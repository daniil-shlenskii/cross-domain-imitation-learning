import os
import time
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class ToyEnv(gym.Env):
    x_start: float = 0.

    def __init__(self, y: float = 0., render_mode: Optional[str] = None):
        self.y = y
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, y]), high=np.array([np.inf, y])
        )
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,))

        self._location = np.array([self.x_start, self.y])

        if render_mode is not None:
            save_dir = ".render_toy_env"
            if not os.path.exists(save_dir):
                os.mkdir(".render_toy_env")
            self.save_path = f"{save_dir}/figure.png"

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        observation = np.array([self.x_start + np.random.rand(), self.y])
        info = {}

        self._location = observation
        if self.render_mode is not None:
            self.render()
        return observation, info

    def step(self, action: np.float32):
        observation = self._location
        observation[0] += action

        reward = observation[0]

        terminated = False
        truncated = False
        info = {}

        self._location = observation
        if self.render_mode is not None:
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        fig = plt.figure()
        plt.axvline(x=self.x_start, color="g")
        plt.scatter(x=[self._location[0]], y=[self._location[1]])
        plt.savefig(self.save_path)
        plt.close(fig)
        time.sleep(0.001)
