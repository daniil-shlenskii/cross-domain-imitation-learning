import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

from .toy_one_dim_env import ToyOneDimEnv, ToyOneDimEnvShifted


def register_envs():
    gym.register(id="custom_envs/ToyOneDimEnv", entry_point=ToyOneDimEnv)
    gym.register(id="custom_envs/ToyOneDimEnvShifted", entry_point=ToyOneDimEnvShifted)

register_envs()

