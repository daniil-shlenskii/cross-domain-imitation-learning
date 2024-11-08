import jax
import jax.numpy as jnp


def process_policy_discriminator_input(x: jnp.ndarray):
    doubled_b_size, dim = x.shape
    x = x.reshape(2, doubled_b_size // 2, dim).transpose(1, 2, 0).reshape(-1, dim * 2)
    return x

@jax.jit
def encode_observations_jit(encoder, observations):
    return encoder(observations)