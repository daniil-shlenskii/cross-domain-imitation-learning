from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from flax import linen as nn
from nn.networks import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        values = MLP(
            hidden_dims=self.hidden_dims,
            out_dim=1,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )(observations, train=train)
        return values.squeeze(-1)

class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        input_array = jnp.concatenate([observations, actions], axis=-1)
        values = MLP(
            hidden_dims=self.hidden_dims,
            out_dim=1,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )(input_array, train=train)
        return values.squeeze(-1)
