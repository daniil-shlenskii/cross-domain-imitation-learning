from pathlib import Path
from typing import Tuple

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from nn.train_state import TrainState
from utils.types import PRNGKey
from utils.utils import SaveLoadMixin


class Agent(SaveLoadMixin):
    actor: TrainState
    rng: PRNGKey
    _save_attrs: Tuple[str] = ("actor",)

    @classmethod
    def instantiate_actor_module(
        cls,
        actor_module_config: DictConfig,
        *,
        action_space: gym.Space,
        **kwargs,
    ) -> nn.Module:
        action_dim = action_space.sample().shape[-1]

        low, high = None, None
        if np.any(action_space.low != -1) or np.any(action_space.high != 1):
            low, high = action_space.low, action_space.high
        
        return instantiate(
            actor_module_config, action_dim=action_dim, low=low, high=high, **kwargs
        )

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        self.rng, actions = _sample_actions_jit(self.rng, self.actor, observations)
        return np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions_jit(self.actor, observations)
        return np.asarray(actions)

    def eval_log_probs(self, observations: np.ndarray, actions: np.ndarray) -> float:
        return _eval_log_probs_jit(self.actor, observations, actions)

@jax.jit
def _sample_actions_jit(rng: PRNGKey, actor: TrainState, observations: np.ndarray) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    dist = actor(observations)
    return rng, dist.sample(seed=key)

@jax.jit
def _eval_actions_jit(actor: TrainState, observations: np.ndarray) -> np.ndarray:
    dist = actor(observations)
    return dist.mode()

@jax.jit
def _eval_log_probs_jit(actor: TrainState, observations: np.ndarray, actions: np.ndarray) -> float:
    dist = actor(observations)
    return dist.log_probs(actions).mean()
