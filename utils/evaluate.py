from typing import Dict

import gymnasium as gym
import jax
import numpy as np


def evaluate(agent: "Agent", env: gym.Env, num_episodes: int, seed: int=0, return_trajectories: bool=False) -> Dict[str, float]:
    trajs = {"observations": [], "observations_next": []}

    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
    for i in range(num_episodes):
        observation, _, done, truncated = *env.reset(seed=seed+i), False, False
        while not (done or truncated):
            action = agent.eval_actions(observation)
            next_observation, _, done, truncated, _ = env.step(action)
            
            trajs["observations"].append(observation)
            trajs["observations_next"].append(next_observation)
            
            observation = next_observation
    
    stats = {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}
    if return_trajectories:
        trajs["observations"] = np.stack(trajs["observations"])
        trajs["observations_next"] = np.stack(trajs["observations_next"])
        return stats, trajs
    return stats

@jax.jit
def apply_model_jit(model, x):
    return model(x)