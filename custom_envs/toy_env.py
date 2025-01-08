import abc
from typing import Optional

import gymnasium as gym
import numpy as np


class BaseToyEnv(gym.Env, abc.ABC):
    x_start: float = 0.

    def __init__(self, size: float = 10.):
        self.size = size

        self.observation_space = self._get_observation_space()
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,))

        self._x_location = self.x_start

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        x_coord = self.x_start + (np.random.rand() - 0.5) * 2. / self.size

        observation = self._get_observation_with_x_coord(x_coord)
        info = {}

        self._x_location = x_coord
        return observation, info

    def step(self, action: np.float32):
        new_x_coord = self._x_location + action[0] / self.size

        observation = self._get_observation_with_x_coord(new_x_coord)
        reward = new_x_coord if new_x_coord < 1. else 1000.
        terminated = not (-1. < new_x_coord < 1.)
        truncated = False
        info = {}

        self._x_location = new_x_coord
        return observation, reward, terminated, truncated, info

    @abc.abstractmethod
    def _get_observation_space(self):
        pass

    @abc.abstractmethod
    def _get_observation_with_x_coord(self, x_coord: np.ndarray):
        pass

class ToyEnvOneDim(BaseToyEnv):
    def _get_observation_space(self):
        observation_space = gym.spaces.Box(-1., 1., shape=(1,))
        return observation_space

    def _get_observation_with_x_coord(self, x_coord: np.ndarray):
        return np.asarray([x_coord])

class ToyEnvTwoDim(BaseToyEnv):
    def __init__(self, y: float = 0., **kwargs):
        assert -1. <= y <= 1.
        super().__init__(**kwargs)
        self.y = y

    def _get_observation_space(self):
        observation_space = gym.spaces.Box(-1., 1., shape=(2,))
        return observation_space

    def _get_observation_with_x_coord(self, x_coord: np.ndarray):
        return np.asarray([x_coord, self.y])
