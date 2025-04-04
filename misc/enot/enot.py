from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf import DictConfig
from ott.geometry import costs

import wandb
from nn.train_state import TrainState
from utils import (SaveLoadFrozenDataclassMixin, convert_figure_to_array,
                   instantiate_optimizer)
from utils.custom_types import PRNGKey

from .batch_preprocessors import BatchPreprocessor, IdentityPreprocessor
from .costs import L2Cost
from .utils import mapping_scatter


class ENOT(PyTreeNode, SaveLoadFrozenDataclassMixin):
    rng: PRNGKey
    transport: TrainState 
    g_potential: TrainState 
    cost_fn: costs.CostFn
    train_cost_fn: costs.CostFn
    source_batch_preprocessor: BatchPreprocessor
    target_batch_preprocessor: BatchPreprocessor
    expectile: float = struct.field(pytree_node=False)
    expectile_loss_coef: float = struct.field(pytree_node=False)
    target_weight: float = struct.field(pytree_node=False)
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        source_dim: int,
        target_dim: int,
        transport_module_config: DictConfig,
        transport_optimizer_config: DictConfig,
        transport_loss_fn_config: DictConfig,
        g_potential_module_config: DictConfig,
        g_potential_optimizer_config: DictConfig,
        g_potential_loss_fn_config: DictConfig,
        cost_fn_config: DictConfig = None,
        train_cost_fn_config: DictConfig = None,
        source_batch_preprocessor_config: DictConfig=None,
        target_batch_preprocessor_config: DictConfig=None,
        expectile: float = 0.99,
        expectile_loss_coef: float = 2.0,
        target_weight: float = 1.0,
        **kwargs,
    ):
        rng = jax.random.key(seed)
        rng, transport_key, g_potential_key = jax.random.split(rng, 3)

        transport_module = instantiate(transport_module_config, out_dim=target_dim)
        transport_params = transport_module.init(transport_key, np.ones(source_dim))["params"]
        transport = TrainState.create(
            loss_fn=instantiate(transport_loss_fn_config),
            apply_fn=transport_module.apply,
            params=transport_params,
            tx=instantiate_optimizer(transport_optimizer_config),
            info_key="transport",
        )

        g_potential_module = instantiate(g_potential_module_config, out_dim=1, squeeze=True)
        g_potential_params = g_potential_module.init(g_potential_key, np.ones(target_dim))["params"]
        g_potential = TrainState.create(
            loss_fn=instantiate(g_potential_loss_fn_config),
            apply_fn=g_potential_module.apply,
            params=g_potential_params,
            tx=instantiate_optimizer(g_potential_optimizer_config),
            info_key="g_potential",
        )

        cost_fn = instantiate(cost_fn_config, _recursive_=False) if cost_fn_config is not None else L2Cost()
        train_cost_fn = instantiate(train_cost_fn_config, _recursive_=False) if train_cost_fn_config is not None else cost_fn

        if source_batch_preprocessor_config is not None:
            source_batch_preprocessor = instantiate(source_batch_preprocessor_config, dim=source_dim)
        else:
            source_batch_preprocessor = IdentityPreprocessor.create()

        if target_batch_preprocessor_config is not None:
            target_batch_preprocessor = instantiate(target_batch_preprocessor_config, dim=target_dim)
        else:
            target_batch_preprocessor = IdentityPreprocessor.create()

        _save_attrs = kwargs.pop("_save_attrs", (
            "transport",
            "g_potential",
            "cost_fn",
            "source_batch_preprocessor",
            "target_batch_preprocessor",
        ))

        return cls(
            rng=rng,
            transport=transport,
            g_potential=g_potential,
            cost_fn=cost_fn,
            train_cost_fn=train_cost_fn,
            source_batch_preprocessor=source_batch_preprocessor,
            target_batch_preprocessor=target_batch_preprocessor,
            expectile=expectile,
            expectile_loss_coef=expectile_loss_coef,
            target_weight=target_weight,
            _save_attrs=_save_attrs,
            **kwargs,
        )

    @jax.jit
    def __call__(self, source: jnp.ndarray):
        source_encoded = self.source_batch_preprocessor.encode(source)
        target_hat_encoded = self.transport(source_encoded)
        target_hat = self.target_batch_preprocessor.decode(target_hat_encoded)
        return target_hat

    def cost(self, source: jnp.ndarray, target: jnp.ndarray):
        return jax.vmap(self.cost_fn)(
           self.source_batch_preprocessor.encode(source),
           self.target_batch_preprocessor.encode(target)
        )

    def g_potential_val(self, target: jnp.ndarray):
        return self.g_potential(self.target_batch_preprocessor.encode(target))

    def update(self, target: jnp.ndarray, source: jnp.ndarray):
        enot, info, stats_info = _update_jit(
            enot=self,
            target=target,
            source=source,
        )
        return enot, info, stats_info

    def evaluate(self, source: jnp.ndarray, target: jnp.ndarray, convert_to_wandb_type: bool=True):
        target_hat = self(source)

        fig = mapping_scatter(source, target_hat, target)
        if convert_to_wandb_type:
            fig = wandb.Image(convert_figure_to_array(fig), caption="")

        return {
            "enot/mapping_scatter": fig,
            "enot/cost": self.cost(source, target_hat).mean(),
        }

@jax.jit
def _update_jit(
    enot: ENOT,
    target: jnp.ndarray,
    source: jnp.ndarray,
) -> Tuple[ENOT, Dict[str, Any], Dict[str, Any]]:
    # preprocess batches
    new_source_batch_preprocessor, source, _ = enot.source_batch_preprocessor.update(source)
    new_target_batch_preprocessor, target, _ = enot.target_batch_preprocessor.update(target)

    # update cost fn
    target_hat = enot.transport(source)
    new_cost_fn = enot.cost_fn.update(source=source, target=target_hat)
    new_train_cost_fn = enot.train_cost_fn.update(source=source, target=target_hat)
    enot = enot.replace(
        target_batch_preprocessor=new_target_batch_preprocessor,
        source_batch_preprocessor=new_source_batch_preprocessor,
        cost_fn=new_cost_fn,
        train_cost_fn=new_train_cost_fn
    )

    # update transport and g_potential
    new_transport, new_transport_info, new_transport_stats_info = enot.transport.update(
        source=source,
        enot=enot
    )
    target_hat = new_transport_info.pop("target_hat")
    new_g_potential, new_g_potential_info, new_g_potential_stats_info = enot.g_potential.update(
        source=source,
        target=target,
        target_hat=target_hat,
        enot=enot,
    )
    new_enot = enot.replace(
        transport=new_transport,
        g_potential=new_g_potential,
    )
    info = {**new_transport_info, **new_g_potential_info}
    stats_info = {**new_transport_stats_info, **new_g_potential_stats_info}
    return new_enot, info, stats_info
