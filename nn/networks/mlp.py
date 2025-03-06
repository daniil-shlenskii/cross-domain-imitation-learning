import flax.linen as nn
import jax.numpy as jnp

from .utils import default_init


class MLPBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.relu(x)
        return x

class MLP(nn.Module):
    n_blocks: int
    hidden_dim: int
    out_dim: int = None
    squeeze: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False
    ) -> jnp.ndarray:
        for _ in range(self.n_blocks):
            x = MLPBlock(self.hidden_dim)(x)

        if self.out_dim is not None:
            x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        if self.squeeze:
            x = x.squeeze(-1)

        return x

class NegativeMLP(MLP):
    def __call__(
        self, x: jnp.ndarray, train: bool=False
    ) -> jnp.ndarray:
        x = super().__call__(x, train)
        return -jnp.abs(x)
