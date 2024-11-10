import jax
import jax.numpy as jnp


@jax.jit
def process_policy_discriminator_input(x: jnp.ndarray):
    doubled_b_size, dim = x.shape
    x = x.reshape(2, doubled_b_size // 2, dim).transpose(1, 0, 2).reshape(-1, dim * 2)
    return x

@jax.jit
def encode_observation(encoder, observations):
    return encoder(observations)