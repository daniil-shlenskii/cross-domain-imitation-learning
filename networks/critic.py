from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from networks.common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        values = MLP((*self.hidden_dims, 1))(observations)
        return values.squeeze(-1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        input_array = jnp.concatenate([observations, actions], axis=-1)
        values = MLP((*self.hidden_dims, 1))(input_array, training=training)
        return values.squeeze(-1)

class CriticEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    n_modules: int = 2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        critic_ensemble = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_modules,
        )
        q_values = critic_ensemble(self.hidden_dims)(
            observations, actions, training
        )
        return jnp.min(q_values, axis=0)