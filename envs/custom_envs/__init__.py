import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

from .toy_one_dim_env import ToyOneDimEnv, ToyOneDimEnvShifted


def register_envs():
    gym.register(id="custom_envs/ToyOneDimEnv", entry_point=ToyOneDimEnv)
    gym.register(id="custom_envs/ToyOneDimEnvShifted", entry_point=ToyOneDimEnvShifted)
    gym.register(
        id="CustomPointUmaze",
        entry_point="envs.custom_envs.custom_point_umaze:CustomPointUmaze",
        max_episode_steps=300,
        kwargs={
            "reward_type": "dense",
            "continuing_task": True,
            "reset_target": False,
        },
        additional_wrappers=[
            gym.envs.registration.WrapperSpec(
                name="TransformObservation",
                entry_point="gymnasium.wrappers:TransformObservation",
                kwargs={
                    "func": lambda obs: obs["observation"],
                    "observation_space": gym.spaces.Box(-np.inf, np.inf, shape=(4,)),
                },
            )
        ],
    )

register_envs()
