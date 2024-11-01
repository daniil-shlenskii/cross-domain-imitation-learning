from typing import Callable

import jax
import jax.numpy as jnp

from utils.types import PRNGKey


def g_nonsaturating_loss(fake_logits: jnp.ndarray):
    loss = jax.nn.softplus(-fake_logits) 
    return loss

def d_logistic_loss(real_logits: jnp.ndarray, fake_logits: jnp.ndarray):
    real_loss = jax.nn.softplus(-real_logits)
    fake_loss = jax.nn.softplus(fake_logits)
    return real_loss.mean() + fake_loss.mean()

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