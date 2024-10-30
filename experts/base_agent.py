from typing import Tuple, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from nn.train_state import TrainState

from utils.utils import save_pickle, load_pickle

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
    
    def save(self, dir_path: str) -> None:
        dir_path = Path(dir_path)
        for k, v in self.__dict__.items():
            if isinstance(v, TrainState):
                v.save(dir_path / f"{k}.pickle")

    def load(self, dir_path: str) -> None:
        dir_path = Path(dir_path)
        loaded_attrs = []
        for path in dir_path.iterdir():
            attr = str(path).split("/")[-1].split(".")[0]
            if hasattr(self, attr):
                train_state = getattr(self, attr)
                if isinstance(train_state, TrainState):
                    loaded_train_state = train_state.load(path)
                    setattr(self, attr, loaded_train_state)
                    loaded_attrs.append(attr)
        return loaded_attrs

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
