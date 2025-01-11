from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class Identity(nn.Module):
    out_dim: int = None

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False
    ) -> jnp.ndarray:
        y = nn.Dense(1)(x)
        return x

class AffineTransformIdInit(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False,
    ):
        x = x + nn.Dense(self.out_dim, kernel_init=constant(0.))(x)
        return x
