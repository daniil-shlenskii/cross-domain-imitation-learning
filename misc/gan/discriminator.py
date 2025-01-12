from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from nn.train_state import TrainState
from omegaconf.dictconfig import DictConfig

from utils import SaveLoadFrozenDataclassMixin, instantiate_optimizer
from utils.custom_types import PRNGKey


class Discriminator(PyTreeNode, SaveLoadFrozenDataclassMixin):
    rng: PRNGKey
    state: TrainState
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        input_dim: int,
        #
        module_config: DictConfig,
        optimizer_config: DictConfig,
        loss_config: DictConfig = None,
        #
        info_key: str = "discriminator",
        **kwargs,
    ):
        rng = jax.random.key(seed)
        rng, key = jax.random.split(rng)

        module = instantiate(module_config)
        params = module.init(key, jnp.ones(input_dim, dtype=jnp.float32))["params"]

        loss_fn = instantiate(loss_config, _recursive_=False)

        state = TrainState.create(
            loss_fn=loss_fn,
            apply_fn=module.apply,
            params=params,
            tx=instantiate_optimizer(optimizer_config),
            info_key=info_key,
        )

        _save_attrs = kwargs.pop("_save_attrs", ("state",))

        return cls(
            rng=rng,
            state=state,
            _save_attrs=_save_attrs,
            **kwargs
        )

    def update(self, *, real_batch: jnp.ndarray, fake_batch: jnp.ndarray, return_logits: bool=False, **kwargs):
        new_state, info, stats_info = self.state.update(real_batch=real_batch, fake_batch=fake_batch, **kwargs)
        if not return_logits:
            del info["real_logits"], info["fake_logits"]
        return self.replace(state=new_state), info, stats_info

    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.state(x, *args, **kwargs)
