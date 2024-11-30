from typing import Callable

import jax
import jax.numpy as jnp

from utils.types import PRNGKey

##### generator losses #####

def g_nonsaturating_logistic_loss(fake_logits: jnp.ndarray):
    fake_probs = jax.nn.sigmoid(fake_logits)
    loss = -jnp.log(fake_probs)
    return loss

def g_nonsaturating_softplus_loss(fake_logits: jnp.ndarray):
    loss = jax.nn.softplus(-fake_logits)
    return loss


##### discriminator losses #####

def d_logistic_loss(real_logits: jnp.ndarray, fake_logits: jnp.ndarray):
    real_probs = jax.nn.sigmoid(real_logits)
    fake_probs = jax.nn.sigmoid(fake_logits)
    real_loss = -jnp.log(real_probs)
    fake_loss = -jnp.log(1 - fake_probs)
    return (real_loss + fake_loss) * 0.5

def d_softplus_loss(real_logits: jnp.ndarray, fake_logits: jnp.ndarray):
    real_loss = jax.nn.softplus(-real_logits)
    fake_loss = jax.nn.softplus(fake_logits)
    return (real_loss + fake_loss) * 0.5

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
    return norms
