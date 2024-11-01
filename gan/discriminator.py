from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from flax.struct import PyTreeNode

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from nn.train_state import TrainState
from utils.utils import instantiate_optimizer
from utils.types import Params, PRNGKey

from gan.losses import d_logistic_loss, gradient_penalty


class Discriminator(PyTreeNode):
    state: TrainState

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        input_sample: jnp.ndarray,
        #
        module_config: DictConfig,
        optimizer_config: DictConfig,
    ):
        key = jax.random.key(seed)
        module = instantiate(module_config)
        params = module.init(key, input_sample)["params"]
        state = TrainState.create(
            loss_fn=_discr_loss_fn,
            apply_fn=module.apply,
            params=params,
            tx=instantiate_optimizer(optimizer_config),
            info_key="discriminator",
        )
        return cls(state=state)

    def update(self, *, real_batch: jnp.ndarray, fake_batch: jnp.ndarray):
        self.state, info, stats_info = _update_jit(real_batch, fake_batch, self.state)
        return info, stats_info
    
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.state(x, *args, **kwargs)
    
@jax.jit
def _update_jit(
    real_batch: jnp.ndarray,
    fake_batch: jnp.ndarray,
    state: TrainState
) -> Tuple[TrainState, Dict, Dict]:
    new_state, info, stats_info = state.update(real_batch, fake_batch)
    return new_state, info, stats_info

def _discr_loss_fn(
    params: Params,
    state: TrainState,
    real_batch: jnp.ndarray,
    fake_batch: jnp.ndarray,
    gradient_penalty_coef: float,
    key: PRNGKey,
):
    real_logits = state.apply_fn({"params": params}, real_batch)
    fake_logits = state.apply_fn({"params": params}, fake_batch)
    loss = d_logistic_loss(real_logits=real_logits, fake_logits=fake_logits)

    disc_grad_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x))
    penalty = gradient_penalty(key=key, real_batch=real_batch, fake_batch=fake_batch, discriminator_grad_fn=disc_grad_fn)

    return loss + penalty * gradient_penalty_coef