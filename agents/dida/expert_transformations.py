import abc
from copy import deepcopy
from typing import Dict

import jax
import jax.numpy as jnp

from utils.types import BufferState, PRNGKey


class ObservationsTransform(abc.ABC):
    _observations_keys = (
        "observations",
        "observations_next"
    )

    @property
    def key(self) -> PRNGKey:
        if not hasattr(self, "rng"):
            self.rng = jax.random.key(0)
        self.rng, key = jax.random.split(self.rng)
        return key

    def __call__(self, state: BufferState) -> BufferState:
        state_exp = state.experience
        new_state_exp = deepcopy(state_exp)
        new_state_exp = self.update_state_exp(new_state_exp)
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
            state_exp[k] = add_noise_jit(
                key=self.key,
                x=state_exp[k],
                mu=self.mu,
                sigma=self.sigma
            )
        return state_exp

def add_noise_jit(key: PRNGKey, x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    noise = jax.random.normal(key, shape=x.shape)
    return x + noise * sigma + mu

