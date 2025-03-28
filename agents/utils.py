from typing import Dict

import gymnasium as gym
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

TRAJ_KEYS = [
    "observations",
    "actions",
    "rewards",
    "dones",
    "truncated",
    "observations_next",
]

def instantiate_agent(config: DictConfig, *, env: gym.Env, **kwargs):
    observation_space = env.observation_space
    action_space = env.action_space

    observation= observation_space.sample()
    action = action_space.sample()
    if not isinstance(observation, np.ndarray): # UMaze case
        observation = observation["observation"]

    agent =  instantiate(
        config,
        observation_dim=observation.shape[-1],
        action_dim=action.shape[-1],
        low=None,
        high=None,
        **kwargs,
        _recursive_=False,
    )

    return agent

def evaluate(
    agent: "Agent",
    env: gym.Env,
    num_episodes: int,
    seed: int = 0,
    return_trajectories: bool = False,
) -> Dict[str, float]:
    trajs = {traj_key: [] for traj_key in TRAJ_KEYS}

    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
    for i in range(num_episodes):
        observation, _, done, truncated = *env.reset(seed=seed+i), False, False
        if not isinstance(observation, np.ndarray): # UMaze case
            observation = observation["observation"]

        while not (done or truncated):
            action = agent.eval_actions(observation)
            observation_next, reward, done, truncated, _ = env.step(action)
            if not isinstance(observation_next, np.ndarray): # UMaze case
                observation_next = observation_next["observation"]

            trajs["observations"].append(observation)
            trajs["actions"].append(action)
            trajs["rewards"].append(reward)
            trajs["dones"].append(done)
            trajs["truncated"].append(done or truncated)
            trajs["observations_next"].append(observation_next)

            observation = observation_next

    stats = {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}
    if return_trajectories:
        for k, v in trajs.items():
            trajs[k] = np.stack(v)
        return stats, trajs
    return stats
