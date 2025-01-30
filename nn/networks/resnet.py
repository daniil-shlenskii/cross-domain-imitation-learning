import flax.linen as nn
import jax.numpy as jnp

from .utils import default_init, he_normal_init


class ResidualBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim * 4, kernel_init=he_normal_init())(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=he_normal_init())(x)
        return res + x


class ResNet(nn.Module):
    n_blocks: int
    hidden_dim: int
    out_dim: int = None
    squeeze: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool=False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.n_blocks):
            x = ResidualBlock(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)

        if self.out_dim is not None:
            x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        if self.squeeze:
            x = x.squeeze(-1)

        return x
