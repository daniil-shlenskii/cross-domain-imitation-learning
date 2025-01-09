import abc
import os
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class BaseToyEnv(gym.Env, abc.ABC):
    x_start: float = 0.
    render_dir: str = "._render_toy_env"
    render_file: str = f"{render_dir}/figure.png"

    def __init__(self, size: float = 10., render_mode: Optional[str] = None, render_every_frame: int = 1):
        self.size = size
        self.render_mode = render_mode
        self.render_every_frame = render_every_frame
        self._frame_idx = 0

        self.observation_space = self._get_observation_space()
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,))

        self._x_location = self.x_start
        self._terminated = False

        if render_mode is not None:
            if not os.path.exists(self.render_dir):
                os.mkdir(self.render_dir)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        x_coord = self.x_start + (np.random.rand() - 0.5) * 2. / self.size

        observation = self._get_observation_with_x_coord(x_coord)
        info = {}

        self._x_location = x_coord

        if self.render_mode is not None:
            self.render()
        return observation, info

    def step(self, action: np.float32):
        new_x_coord = self._x_location + action[0] / self.size

        observation = self._get_observation_with_x_coord(new_x_coord)
        reward = new_x_coord if new_x_coord < 1. else 1000.
        terminated = not (-1. < new_x_coord < 1.)
        truncated = False
        info = {}

        self._x_location = new_x_coord
        self._terminated = terminated

        if self.render_mode is not None:
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self._terminated:
            plt.clf()
            self._frame_idx = 0
        if self._frame_idx % self.render_every_frame == 0:
            y = self.y if hasattr(self, "y") else 0.
            x = self._x_location
            plt.scatter(x=[x], y=[y], c="b", alpha=np.clip(0.1 + 0.9 * 0.3 *self._frame_idx / self.size, -1, 1.))
            plt.axvline(x=self.x_start, color="g")
            plt.axvline(x=1., color="r")
            plt.savefig(self.render_file)
        self._frame_idx += 1


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
