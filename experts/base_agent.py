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
    save_attrs: Sequence[str] = ("step", "params", "opt_state")

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
            if type(v) == TrainState:
                v.save(dir_path / f"{k}.pickle")

    def load(self, dir_path: str) -> None:
        dir_path = Path(dir_path)
        loaded_attrs = []
        for path in dir_path.iterdir():
            attr = str(path).split("/")[-1].split(".")[0]
            if hasattr(self, attr):
                train_state = getattr(self, attr)
                loaded_train_state = train_state.load(path)
                setattr(self, attr, loaded_train_state)
                loaded_attrs.append(attr)
        return loaded_attrs

    # def save(self, path: str):
    #     train_states = {}
    #     for k, v in self.__dict__.items():
    #         if type(v) != TrainState:
    #             continue
    #         train_states[k] = {
    #             attr: v.__dict__[attr]
    #             for attr in self.save_attrs
    #         }
    #     save_pickle(train_states, path)

    # def load(self, path: str):
    #     train_states = load_pickle(path)
    #     for train_state_name, train_state_dict in train_states.items():
    #         if train_state_name not in self.__dict__:
    #             raise ValueError(f"Incompatible attribute name. Agent does not have attribute {train_state_name}.")
    #         for attr in self.save_attrs:
    #             assert attr in self.__dict__[train_state_name].__dict__
    #             self.__dict__[train_state_name].__dict__[attr] = train_state_dict[attr]
    #     return train_states.keys()

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
