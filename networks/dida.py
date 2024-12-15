from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant

from networks.common import MLP


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class AffineTransform(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False,
    ):
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x

class SkipMLPAffineTransform(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int = None
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __cal__(
        self, x: jnp.ndarray, train: bool=False,           
    ):
        res = MLP(
            hidden_dims=self.hidden_dims,
            out_dim=None,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
        )
        res = nn.Dense(x.shape[-1], kernel_init=constant(0.))(res)

        x = x + res

        x = AffineTransform(out_dim=self.out_dim)(x)
        return x
