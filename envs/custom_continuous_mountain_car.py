import math
from typing import Optional

import numpy as np
from gymnasium.envs.classic_control import Continuous_MountainCarEnv


class Custom_Continuous_MountainCarEnv(Continuous_MountainCarEnv):
    def __init__(
        self,
        *,
        time_discretization: float = 1.,
        render_mode: str = None,
        goal_velocity: float = 0.
    ):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)
        self.h = time_discretization

    def step(self, action: np.ndarray):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += (force * self.power - 0.0025 * math.cos(3 * position)) * self.h**2
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity * self.h
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = (position - self.goal_position) * self.h

        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self.state, reward, terminated, False, {}