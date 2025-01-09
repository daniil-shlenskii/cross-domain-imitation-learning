from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from flax import linen as nn

from networks.common import MLP


class Discriminator(nn.Module):
    hidden_dims: Sequence[int] = tuple()
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        values = MLP(
            hidden_dims=self.hidden_dims,
            out_dim=1,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )(x, train=train)
        return values.squeeze(-1)
