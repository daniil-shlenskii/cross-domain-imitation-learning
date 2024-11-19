import functools
from pathlib import Path
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from gan.discriminator import Discriminator
from gan.losses import g_softplus_loss
from hydra.utils import instantiate
from nn.train_state import TrainState
from omegaconf.dictconfig import DictConfig
from utils.types import Params
from utils.utils import SaveLoadFrozenDataclassMixin, instantiate_optimizer


class Generator(PyTreeNode, SaveLoadFrozenDataclassMixin):
    state: TrainState
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        input_dim: int,
        output_dim:  int,
        #
        module_config: DictConfig,
        optimizer_config: DictConfig,
        loss_fn_config: DictConfig = None,
        #
        info_key: str = "generator",
        **kwargs,
    ):
        module_config.hidden_dims.append(output_dim)

        key = jax.random.key(seed)
        module = instantiate(module_config)
        params = module.init(key, jnp.ones(input_dim, dtype=jnp.float32))["params"]

        if loss_fn_config is None:
            loss_fn = g_softplus_loss
        else:
            loss_fn = instantiate(loss_fn_config)

        state = TrainState.create(
            loss_fn=loss_fn,
            apply_fn=module.apply,
            params=params,
            tx=instantiate_optimizer(optimizer_config),
            info_key=info_key,
        )

        _save_attrs = kwargs.pop("_save_attrs", ("state",))

        return cls(
            state=state,
            _save_attrs=_save_attrs,
            **kwargs
        )

    def update(self, *, batch: Any, **kwargs):
        new_state, info, stats_info = _update_jit(
            batch=batch,
            state=self.state,
            **kwargs
        )
        return self.replace(state=new_state), info, stats_info
    
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.state(x, *args, **kwargs)
    
def _update_jit(
    batch: Any,
    state: TrainState,
    **kwargs,
):
    new_state, info, stats_info = state.update(
        batch=batch,
        **kwargs,
    )
    return new_state, info, stats_info
