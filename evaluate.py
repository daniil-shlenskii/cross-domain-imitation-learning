from typing import Dict

import gymnasium as gym
import numpy as np

from experts.base_agent import Agent


def evaluate(agent: Agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
    for _ in range(num_episodes):
        observation, _, done, truncated = *env.reset(), False, False
        while not (done or truncated):
            action = agent.sample_actions(observation)
            observation, _, done, truncated, _ = env.step(action)


    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}