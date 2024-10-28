from typing import Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool=False
    ) -> jnp.ndarray:
        for hid_dim in self.hidden_dims:
            x = nn.Dense(hid_dim, kernel_init=default_init())(x)
            x = self.activation(x)
            if self.dropout_rate is not None:
                x = nn.Dropout(reate=self.dropout_rate)(
                    x, deterministic=not training
                )
        return x