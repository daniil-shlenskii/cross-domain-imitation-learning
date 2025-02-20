from typing import (Any, Callable, Dict, Iterator, List, Literal, Optional,
                    Sequence, Tuple, Union)

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf import DictConfig
from ott.geometry import costs

from nn.train_state import TrainState
from utils import instantiate_optimizer
from utils.custom_types import DataType, PRNGKey

DType = Union[jnp.ndarray, DataType]


class ENOT(PyTreeNode):
    rng: PRNGKey
    transport: TrainState 
    g_potential: TrainState 
    cost_fn: costs.CostFn = struct.field(pytree_node=False)
    expectile: float = struct.field(pytree_node=False)
    expectile_loss_coef: float = struct.field(pytree_node=False)
    target_weight: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        data_dim: int,
        transport_module_config: DictConfig,
        transport_optimizer_config: DictConfig,
        transport_loss_fn_config: DictConfig,
        g_potential_module_config: DictConfig,
        g_potential_optimizer_config: DictConfig,
        g_potential_loss_fn_config: DictConfig,
        cost_fn_config: DictConfig = None,
        expectile: float = 0.99,
        expectile_loss_coef: float = 1.0,
        target_weight: float = 1.0
    ):
        rng = jax.random.key(seed)
        rng, transport_key, g_potential_key = jax.random.split(rng, 3)

        transport_module = instantiate(transport_module_config, out_dim=data_dim)
        transport_params = transport_module.init(transport_key, np.ones(data_dim))["params"]
        transport = TrainState.create(
            loss_fn=instantiate(transport_loss_fn_config),
            apply_fn=transport_module.apply,
            params=transport_params,
            tx=instantiate_optimizer(transport_optimizer_config),
            info_key="transport",
        )

        g_potential_module = instantiate(g_potential_module_config, out_dim=1, squeeze=True)
        g_potential_params = g_potential_module.init(g_potential_key, np.ones(data_dim))["params"]
        g_potential = TrainState.create(
            loss_fn=instantiate(g_potential_loss_fn_config),
            apply_fn=g_potential_module.apply,
            params=g_potential_params,
            tx=instantiate_optimizer(g_potential_optimizer_config),
            info_key="g_potential",
        )

        cost_fn = instantiate(cost_fn_config) if cost_fn_config is not None else costs.SqEuclidean()
        cost_fn = jax.vmap(cost_fn)

        return cls(
            rng=rng,
            transport=transport,
            g_potential=g_potential,
            cost_fn=cost_fn,
            expectile=expectile,
            expectile_loss_coef=expectile_loss_coef,
            target_weight=target_weight,
        )

    @jax.jit
    def __call__(self, source: DType):
        return self.transport(source)

    def update(self, target: DType, source: DType):
        enot, info, stats_info = _update_jit(
            enot=self,
            target=target,
            source=source,
        )
        return enot, info, stats_info

@jax.jit
def _update_jit(
    enot: ENOT,
    target: DType,
    source: DType,
) -> Tuple[ENOT, Dict[str, Any], Dict[str, Any]]:
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
