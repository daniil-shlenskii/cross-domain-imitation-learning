from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from networks.common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        values = MLP((*self.hidden_dim, 1))(observations)
        return values.squeeze(-1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        input_array = jnp.concatenate([observations, actions], axis=-1)
        values = MLP((*self.hidden_dim, 1))(input_array)
        return values.squeeze(-1)
