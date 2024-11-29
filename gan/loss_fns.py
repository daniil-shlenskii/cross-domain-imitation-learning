import functools
from typing import Callable, Union

import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from gan.base_losses import d_logistic_loss as base_d_logistic_loss
from gan.base_losses import d_softplus_loss as base_d_softplus_loss
from gan.base_losses import \
    g_nonsaturating_logistic_loss as base_g_nonsaturating_logistic_loss
from gan.base_losses import \
    g_nonsaturating_softplus_loss as base_g_nonsaturating_softplus_loss
from gan.base_losses import gradient_penalty
from nn.train_state import TrainState
from utils.types import Params, PRNGKey

#### generator losses (simple) ####

def _g_loss_simple(
    params: Params,
    state: TrainState,
    batch: jnp.ndarray,
    discriminator: "Discriminator",
    #
    loss_fn: Callable,
):
    fake_batch = state.apply_fn({"params": params}, batch, train=True)
    fake_logits = discriminator(fake_batch)
    loss = loss_fn(fake_logits)

    info = {
        f"{state.info_key}_loss": loss,
        "generations": fake_batch
    }
    return loss, info

g_logistic_loss = functools.partial(
    _g_loss_simple,
    loss_fn=base_g_nonsaturating_logistic_loss
)

g_softplus_loss = functools.partial(
    _g_loss_simple,
    loss_fn=base_g_nonsaturating_softplus_loss
)

#### discriminator losses (simple) ####

def _d_loss_simple(
    params: Params,
    state: TrainState,
    real_batch: jnp.ndarray,
    fake_batch: jnp.ndarray,
    #
    loss_fn: Callable,
):
    real_logits = state.apply_fn({"params": params}, real_batch, train=True)
    fake_logits = state.apply_fn({"params": params}, fake_batch, train=True)
    loss = loss_fn(real_logits=real_logits, fake_logits=fake_logits)

    info = {
        f"{state.info_key}_loss": loss,
        "real_logits": real_logits,
        "fake_logits": fake_logits,
    }
    return loss, info

d_logistic_loss = functools.partial(
    _d_loss_simple,
    loss_fn=base_d_logistic_loss
)

d_softplus_loss = functools.partial(
    _d_loss_simple,
    loss_fn=base_d_softplus_loss
)

#### discriminator losses with gradient penalty ####

class GradientPenaltyDecorator:
    def __init__(
        self,
        d_loss_fn: Union[Callable, DictConfig],
        gradient_penalty_coef: float
    ):
        if not isinstance(d_loss_fn, Callable):
            d_loss_fn = instantiate(d_loss_fn)
        self.d_loss_fn = d_loss_fn
        self.penalty_coef = gradient_penalty_coef
        
    @property
    def key(self):
        if not hasattr(self, "rng"):
            self.rng = jax.random.key(0)
        self.rng, key = jax.random.split(self.rng) 
        return key

    def __call__(
        self,
        params: Params,
        state: TrainState,
        real_batch: jnp.ndarray,
        fake_batch: jnp.ndarray,
        **kwargs,
    ):
        d_loss, info = self.d_loss_fn(
            params=params,
            state=state,
            real_batch=real_batch,
            fake_batch=fake_batch,
            **kwargs
        )

        disc_grad_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x, train=True))
        penalty = gradient_penalty(key=self.key, real_batch=real_batch, fake_batch=fake_batch, discriminator_grad_fn=disc_grad_fn)

        loss_with_gp = d_loss + self.penalty_coef * penalty

        info.update({
            "loss_with_gp": loss_with_gp,
            "gradient_penalty": penalty
        }) 

        return loss_with_gp, info
