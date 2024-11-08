from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from gan.discriminator import Discriminator
from gan.losses import g_nonsaturating_loss
from nn.train_state import TrainState
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
        input_sample: jnp.ndarray,
        output_dim:  int,
        #
        module_config: DictConfig,
        optimizer_config: DictConfig,
        #
        info_key: str = "generator",
        **kwargs,
    ):
        module_config.hidden_dims.append(output_dim)

        key = jax.random.key(seed)
        module = instantiate(module_config)
        params = module.init(key, input_sample)["params"]
        state = TrainState.create(
            loss_fn=_gan_loss_fn,
            apply_fn=module.apply,
            params=params,
            tx=instantiate_optimizer(optimizer_config),
            info_key=info_key,
        )

        return cls(
            state=state,
            _save_attrs=kwargs.pop("_save_attrs", "state"),
            **kwargs
        )

    def update(self, *, batch: jnp.ndarray, discriminator: Discriminator):
        new_state, info, stats_info = _update_jit(
            batch=batch,
            state=self.state,
            discriminator=discriminator,
        )
        return self.replace(state=new_state), info, stats_info
    
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.state(x, *args, **kwargs)
    
@jax.jit
def _update_jit(batch: jnp.ndarray, state: TrainState, discriminator: Discriminator):
    new_state, info, stats_info = state.update(batch=batch, discriminator=discriminator)
    return new_state, info, stats_info

def _gan_loss_fn(
    params: Params,
    state: TrainState,
    batch: jnp.ndarray,
    discriminator: Discriminator,
):
    fake_batch = state.apply_fn({"params": params}, batch, train=True)
    fake_logits = discriminator(fake_batch)
    loss = g_nonsaturating_loss(fake_logits)

    info = {
        f"{state.info_key}_loss": loss,
        "fake_batch": fake_batch
    }
    return loss, info