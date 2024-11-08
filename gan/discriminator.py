from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from gan.losses import d_logistic_loss, gradient_penalty
from nn.train_state import TrainState
from utils.types import Params, PRNGKey
from utils.utils import SaveLoadFrozenDataclassMixin, instantiate_optimizer


class Discriminator(PyTreeNode, SaveLoadFrozenDataclassMixin):
    rng: PRNGKey
    state: TrainState
    gradient_penalty_coef: float
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        input_sample: jnp.ndarray,
        #
        module_config: DictConfig,
        optimizer_config: DictConfig,
        #
        gradient_penalty_coef: float = 1.,
        #
        info_key: str = "discriminator",
        **kwargs,
    ):
        rng = jax.random.key(seed)
        rng, key = jax.random.split(rng)

        module = instantiate(module_config)
        params = module.init(key, input_sample)["params"]
        state = TrainState.create(
            loss_fn=_discr_loss_fn,
            apply_fn=module.apply,
            params=params,
            tx=instantiate_optimizer(optimizer_config),
            info_key=info_key,
        )

        if "_save_attrs" not in kwargs:
            kwargs["_save_attrs"] = ("state",)

        return cls(
            rng=rng,
            state=state,
            gradient_penalty_coef=gradient_penalty_coef,
            **kwargs
        )

    def update(self, *, real_batch: jnp.ndarray, fake_batch: jnp.ndarray, **kwargs):
        (
            new_rng,
            new_state,
            info,
            stats_info
        ) = _update_jit(
            real_batch=real_batch,
            fake_batch=fake_batch,
            state=self.state,
            gradient_penalty_coef=self.gradient_penalty_coef,
            rng=self.rng,
            **kwargs
        )
        return self.replace(rng=new_rng, state=new_state), info, stats_info
    
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.state(x, *args, **kwargs)
    
@jax.jit
def _update_jit(
    rng: PRNGKey,
    real_batch: jnp.ndarray,
    fake_batch: jnp.ndarray,
    state: TrainState,
    gradient_penalty_coef: float,
    **kwargs,
) -> Tuple[TrainState, Dict, Dict]:
    new_rng, key = jax.random.split(rng)
    new_state, info, stats_info = state.update(
        key=key,
        real_batch=real_batch,
        fake_batch=fake_batch,
        gradient_penalty_coef=gradient_penalty_coef,
        **kwargs
    )
    return new_rng, new_state, info, stats_info

def _discr_loss_fn(
    params: Params,
    state: TrainState,
    real_batch: jnp.ndarray,
    fake_batch: jnp.ndarray,
    gradient_penalty_coef: float,
    key: PRNGKey,
):
    real_logits = state.apply_fn({"params": params}, real_batch, train=True)
    fake_logits = state.apply_fn({"params": params}, fake_batch, train=True)
    loss = d_logistic_loss(real_logits=real_logits, fake_logits=fake_logits)

    disc_grad_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x, train=True))
    penalty = gradient_penalty(key=key, real_batch=real_batch, fake_batch=fake_batch, discriminator_grad_fn=disc_grad_fn)

    info = {
        f"{state.info_key}_loss": loss,
        f"{state.info_key}_gradient_penalty": penalty
    }
    return loss + penalty * gradient_penalty_coef, info
