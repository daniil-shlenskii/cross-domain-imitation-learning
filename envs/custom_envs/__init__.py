import gymnasium as gym

from .toy_one_dim_env import ToyOneDimEnv, ToyOneDimEnvShifted

gym.register(id="custom_env/ToyOneDimEnv", entry_point=ToyOneDimEnv)
gym.register(id="custom_env/ToyOneDimEnvShifted", entry_point=ToyOneDimEnvShifted)
