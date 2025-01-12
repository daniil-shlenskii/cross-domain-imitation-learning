import functools
from typing import Callable

import jax
import jax.numpy as jnp

from utils.custom_types import PRNGKey

##### generator losses #####

def reduce_g_loss_decorator(func: Callable):
    @functools.wraps(func)
    def wrapped_loss(*args, reduce=True, **kwargs):
        loss = func(*args, **kwargs)
        if reduce:
            return loss.mean()
        return loss
    return wrapped_loss

@reduce_g_loss_decorator
def g_nonsaturating_logistic_loss(fake_logits: jnp.ndarray):
    fake_probs = jax.nn.sigmoid(fake_logits)
    loss = -jnp.log(fake_probs)
    return loss

@reduce_g_loss_decorator
def g_nonsaturating_softplus_loss(fake_logits: jnp.ndarray):
    loss = jax.nn.softplus(-fake_logits)
    return loss


##### discriminator losses #####

def reduce_d_loss_decorator(func: Callable):
    @functools.wraps(func)
    def wrapped_loss(*args, reduce=True, **kwargs):
        real_loss, fake_loss = func(*args, **kwargs)
        if reduce:
            real_loss = real_loss.mean()
            fake_loss = fake_loss.mean()
            return (real_loss + fake_loss) * 0.5
        return real_loss, fake_loss
    return wrapped_loss

@reduce_d_loss_decorator
def d_logistic_loss(real_logits: jnp.ndarray, fake_logits: jnp.ndarray):
    real_probs = jax.nn.sigmoid(real_logits)
    fake_probs = jax.nn.sigmoid(fake_logits)
    real_loss = -jnp.log(real_probs)
    fake_loss = -jnp.log(1 - fake_probs)
    return real_loss, fake_loss

@reduce_d_loss_decorator
def d_softplus_loss(real_logits: jnp.ndarray, fake_logits: jnp.ndarray):
    real_loss = jax.nn.softplus(-real_logits)
    fake_loss = jax.nn.softplus(fake_logits)
    return real_loss, fake_loss

def gradient_penalty(
    key: PRNGKey,
    real_batch: jnp.ndarray,
    fake_batch: jnp.ndarray,
    discriminator_grad_fn: Callable,
):
    t = jax.random.uniform(key, shape=(real_batch.shape[0], 1))
    interpolated_batch = real_batch * t + fake_batch * (1 - t)
    grads = jax.vmap(discriminator_grad_fn)(interpolated_batch)
    norms = jax.vmap(jnp.linalg.norm)(grads)
    return norms.mean()
