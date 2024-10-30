from typing import Dict

import gymnasium as gym
import numpy as np

from experts.base_agent import Agent


def evaluate(agent: Agent, env: gym.Env, num_episodes: int, seed: int=0) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
    for i in range(num_episodes):
        observation, _, done, truncated = *env.reset(seed=seed+i), False, False
        while not (done or truncated):
            action = agent.eval_actions(observation)
            observation, _, done, truncated, _ = env.step(action)

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}