from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from networks.common import MLP


class DomainEncoderNet(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int = None
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False
    ) -> jnp.ndarray:
        x = MLP(self.hidden_dims, self.out_dim, self.activation, self.dropout_rate)(x)
        return jax.nn.sigmoid(x)
