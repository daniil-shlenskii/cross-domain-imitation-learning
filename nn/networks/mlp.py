from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from .utils import default_init


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int = None
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None
    squeeze: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False
    ) -> jnp.ndarray:
        for hid_dim in self.hidden_dims:
            x = nn.Dense(hid_dim, kernel_init=default_init())(x)
            x = self.activation(x)
            if self.dropout_rate is not None:
                x = nn.Dropout(reate=self.dropout_rate)(
                    x, deterministic=not train
                )
        if self.out_dim is not None:
            x = nn.Dense(self.out_dim, kernel_init=default_init())(x)

        if self.squeeze:
            x = x.squeeze(-1)

        return x
