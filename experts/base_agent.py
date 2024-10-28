from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from utils.types import PRNGKey


class Agent:
    actor: TrainState
    _rng: PRNGKey

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        self._rng, actions = _sample_actions_jit(self._rng, self.actor, observations)
        return np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions_jit(self.actor, observations)
        return np.asarray(actions)

    def eval_log_probs(self, observations: np.ndarray, actions: np.ndarray) -> float:
        return _eval_log_probs_jit(self.actor, observations, actions)
    

@jax.jit
def _sample_actions_jit(rng: PRNGKey, actor: TrainState, observations: np.ndarray) -> Tuple[PRNGKey, jnp.ndarray]:
    _rng, key = jax.random.split(rng)
    dist = actor(observations)
    return _rng, dist.sample(seed=key)

@jax.jit
def _eval_actions_jit(actor: TrainState, observations: np.ndarray) -> np.ndarray:
    dist = actor(observations)
    return dist.mean()

@jax.jit
def _eval_log_probs_jit(actor: TrainState, observations: np.ndarray, actions: np.ndarray) -> float:
    dist = actor(observations)
    return dist.log_probs(actions).mean()
