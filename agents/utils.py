from typing import Dict

import gymnasium as gym
import numpy as np


def evaluate(
    agent: "Agent",
    env: gym.Env,
    num_episodes: int,
    seed: int = 0,
    return_trajectories: bool = False,
) -> Dict[str, float]:
    traj_keys = [
        "observations",
        "actions",
        "rewards",
        "dones",
        "observations_next",
    ]
    trajs = {traj_key: [] for traj_key in traj_keys}

    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)
    for i in range(num_episodes):
        observation, _, done, truncated = *env.reset(seed=seed+i), False, False
        while not (done or truncated):
            action = agent.eval_actions(observation)
            next_observation, reward, done, truncated, _ = env.step(action)

            trajs["observations"].append(observation)
            trajs["actions"].append(action)
            trajs["rewards"].append(reward)
            trajs["dones"].append(done or truncated)
            trajs["observations_next"].append(next_observation)

            observation = next_observation

    stats = {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}
    if return_trajectories:
        for k, v in trajs.items():
            trajs[k] = np.stack(v)
        return stats, trajs
    return stats
