import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant


class AffineTransform(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train=False):
        x = x + nn.Dense(self.out_dim, kernel_init=constant(0.))(x)
        return x

