from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

import functools

from utils.types import DataType, PRNGKey


class Agent:
    actor: TrainState
    critic: TrainState
    _rng: PRNGKey

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        self._rng, actions = self._sample_actions_jit(observations)
        return np.asarray(actions)

    @functools.partial(jax.jit, static_argnames="self")
    def _sample_actions_jit(self, observations: np.ndarray) -> Tuple[PRNGKey, jnp.ndarray]:
        _rng, key = jax.random.split(self._rng)
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        return _rng, dist.sample(seed=key)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = self._eval_actions_jit(jnp.asarray(observations))
        return np.asarray(actions)
    
    @functools.partial(jax.jit, static_argnames="self")
    def _eval_actions_jit(self, observations: np.ndarray) -> np.ndarray:
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        return dist.mean()

    def eval_log_probs(self, observations: np.ndarray, actions: np.ndarray) -> float:
        return self._eval_log_prob_jit(observations, actions)
    
    @functools.partial(jax.jit, static_argnames="self")
    def _eval_log_probs_jit(self, observations: np.ndarray, actions: np.ndarray) -> float:
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        return dist.log_probs(actions).mean()
