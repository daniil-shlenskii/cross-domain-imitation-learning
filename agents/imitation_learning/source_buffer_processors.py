import abc
from copy import deepcopy
from typing import Dict

import jax
import jax.numpy as jnp

from utils.custom_types import BufferState, PRNGKey


class ObservationsTransform(abc.ABC):
    rng: PRNGKey = jax.random.key(0)
    _observations_keys = (
        "observations",
        "observations_next"
    )

    @property
    def key(self) -> PRNGKey:
        self.rng, key = jax.random.split(self.rng)
        return key

    def __call__(self, state: BufferState) -> BufferState:
        state_exp = state.experience
        state_exp = deepcopy(state_exp)
        new_state_exp = self.update_state_exp(state_exp)
        new_state = state.replace(experience=new_state_exp)
        return new_state

    @abc.abstractmethod
    def update_state_exp(self, state_exp: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        pass

class GaussianTransform(ObservationsTransform):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def update_state_exp(self, state_exp: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        for k in self._observations_keys:
            state_exp[k] = _add_noise_jit(
                key=self.key,
                x=state_exp[k],
                mu=self.mu,
                sigma=self.sigma
            )
        return state_exp

@jax.jit
def _add_noise_jit(key: PRNGKey, x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    noise = jax.random.normal(key, shape=x.shape)
    return x + noise * sigma + mu

class AddToSecondCoordinate(ObservationsTransform):
    def __init__(self, add_value: float=1.):
        self.add_value = add_value
    def update_state_exp(self, state_exp: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        for k in self._observations_keys:
            state_exp[k] = state_exp[k].at[0, :, 1].add(self.add_value)
        return state_exp
