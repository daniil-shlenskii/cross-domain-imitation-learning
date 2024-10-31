from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from flax import linen as nn

from networks.common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        values = MLP(
            (*self.hidden_dims, 1), self.activation, self.dropout_rate
        )(observations)
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
            (*self.hidden_dims, 1), self.activation, self.dropout_rate
        )(input_array, train=train)
        return values.squeeze(-1)

class CriticEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None
    n_modules: int = 2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        critic_ensemble = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_modules,
        )
        q_values = critic_ensemble(
            self.hidden_dims, self.activation, self.dropout_rate
        )(observations, actions, train)
        return q_values