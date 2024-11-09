from pathlib import Path
from typing import Tuple

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from nn.train_state import TrainState
from utils.types import PRNGKey
from utils.utils import SaveLoadFrozenDataclassMixin


class Agent(PyTreeNode, SaveLoadFrozenDataclassMixin):
    rng: PRNGKey
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    def sample_actions(self, key: PRNGKey, observations: np.ndarray,) -> np.ndarray:
        actions = _sample_actions_jit(key, self.actor, observations)
        return np.asarray(actions)

    def eval_actions(self, observations: np.ndarray,) -> np.ndarray:
        actions = _eval_actions_jit(self.actor, observations)
        return np.asarray(actions)

    def eval_log_probs(self, observations: np.ndarray, actions: np.ndarray) -> float:
        return _eval_log_probs_jit(self.actor, observations, actions)

@jax.jit
def _sample_actions_jit(key: PRNGKey, actor: TrainState, observations: np.ndarray) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor(observations)
    return dist.sample(seed=key)

@jax.jit
def _eval_actions_jit(actor: TrainState, observations: np.ndarray) -> np.ndarray:
    dist = actor(observations)
    return dist.mode()

@jax.jit
def _eval_log_probs_jit(actor: TrainState, observations: np.ndarray, actions: np.ndarray) -> float:
    dist = actor(observations)
    return dist.log_probs(actions).mean()
