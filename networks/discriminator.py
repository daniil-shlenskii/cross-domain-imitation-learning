from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from flax import linen as nn

from networks.common import MLP


class Discriminator(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        values = MLP(
            (*self.hidden_dims, 1), self.activation, self.dropout_rate
        )(x)
        return values.squeeze(-1)
