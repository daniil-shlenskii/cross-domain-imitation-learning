from typing import Optional

import gymnasium as gym
import numpy as np


class ToyEnv(gym.Env):
    x_start: float = 0.

    def __init__(self, y: float = 0., length: int = 10.):
        self.y = y
        self.length = length

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, y]), high=np.array([np.inf, y])
        )
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,))

        self._location = np.array([self.x_start, self.y])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        observation = np.array([self.x_start + np.random.rand() / self.length, self.y])
        info = {}

        self._location = observation
        return observation, info

    def step(self, action: np.float32):
        observation = self._location
        observation[0] += action / self.length

        if observation[0] >= 1.:
            terminated = True
            reward = 0.5 * (self.length - 1.)
        else:
            terminated = False
            reward = observation[0] - 1.

        truncated = False
        info = {}

        self._location = observation
        return observation, reward, terminated, truncated, info
#
# env = ToyEnv()
# observation, _ = env.reset()
# action = env.action_space.sample()
# observation, reward, done, truncated, _ = env.step(action)
