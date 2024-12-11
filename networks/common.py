from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: Optional[bool] = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False
    ) -> jnp.ndarray:
        for i, hid_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hid_dim, kernel_init=default_init())(x)

            if i < len(self.hidden_dims) - 1 or self.activate_final:
                x = self.activation(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(reate=self.dropout_rate)(
                        x, deterministic=not train
                    )
        return x

class ResBlock(nn.Module):
    in_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False
    ) -> jnp.ndarray:
        res = nn.LayerNorm()(x)
        res = self.activation(res)
        res = nn.Dense(self.in_dim * 2, kernel_init=constant(0.), bias_init=constant(0.))(res)

        res = nn.LayerNorm()(res)
        res = self.activation(res)
        res = nn.Dense(self.in_dim, kernel_init=constant(0.), bias_init=constant(0.))(res)

        return x + res

class UNet(nn.Module):
    hidden_dims: Sequence[int]
    n_resblocks_per_dim: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = False,
    ):
        res = nn.Dense(self.hidden_dims[0], kernel_init=default_init())(x)

        inter_feats = [res] 
        for i, (n_resblocks, hidden_dim) in enumerate(zip(self.n_resblocks_per_dim, self.hidden_dims)):
            for _ in range(n_resblocks):
                res = ResBlock(in_dim=hidden_dim, activation=self.activation)(res)
                inter_feats.append(res)
            if i < len(self.hidden_dims) - 1:
                hidden_dim_next = self.hidden_dims[i + 1]
                res = nn.Dense(hidden_dim_next, kernel_init=default_init())(res)

        res = nn.Dense(self.hidden_dims[-1], kernel_init=default_init())(res)

        n_resblocks_per_dim_inv = self.n_resblocks_per_dim[::-1]
        hidden_dims_inv = self.hidden_dims[::-1]
        for i, (n_resblocks, hidden_dim) in enumerate(zip(n_resblocks_per_dim_inv, hidden_dims_inv)):
            for _ in range(n_resblocks):
                skip_connection = inter_feats.pop()
                res = res + skip_connection
                res = ResBlock(in_dim=hidden_dim, activation=self.activation)(res)
                inter_feats.append(res)
            if i < len(hidden_dims_inv) - 1:
                hidden_dim_next = hidden_dims_inv[i + 1]
                res = nn.Dense(hidden_dim_next, kernel_init=default_init())(res)

        res = nn.Dense(x.shape[-1], kernel_init=constant(0.))(res)

        return x + res
