from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from nn.networks import MLP


class MLPSigmoidOnTop(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int = None
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None
    squeeze: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        x = MLP(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            squeeze=self.squeeze,
        )(x, train=train)
        return jax.nn.sigmoid(x)
