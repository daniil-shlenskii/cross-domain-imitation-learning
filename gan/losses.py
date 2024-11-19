import functools
from typing import Callable

import jax
import jax.numpy as jnp
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
    key: PRNGKey,
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

def _d_loss_with_gradient_penalty(
    params: Params,
    state: TrainState,
    real_batch: jnp.ndarray,
    fake_batch: jnp.ndarray,
    gradient_penalty_coef: float,
    key: PRNGKey,
    #
    loss_fn: Callable,
):
    real_logits = state.apply_fn({"params": params}, real_batch, train=True)
    fake_logits = state.apply_fn({"params": params}, fake_batch, train=True)
    loss = loss_fn(real_logits=real_logits, fake_logits=fake_logits)

    disc_grad_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x, train=True))
    penalty = gradient_penalty(key=key, real_batch=real_batch, fake_batch=fake_batch, discriminator_grad_fn=disc_grad_fn)

    info = {
        f"{state.info_key}_loss": loss,
        f"{state.info_key}_gradient_penalty": penalty,
        "real_logits": real_logits,
        "fake_logits": fake_logits,
    }
    return loss + penalty * gradient_penalty_coef, info

d_logistic_loss_with_gradient_penalty = functools.partial(
    _d_loss_with_gradient_penalty,
    loss_fn=base_d_logistic_loss
)

d_softplus_loss_with_gradient_penalty = functools.partial(
    _d_loss_with_gradient_penalty,
    loss_fn=base_d_softplus_loss
)
