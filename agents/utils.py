from typing import Dict

import flashbax as fbx
import gymnasium as gym
import numpy as np

TRAJ_KEYS = [
    "observations",
    "actions",
    "rewards",
    "dones",
    "truncated",
    "observations_next",
]


def get_buffer_state_from_trajectories(trajs):
    # buffer state init
    buffer = fbx.make_item_buffer(
        max_length=len(trajs["observations"]),
        min_length=1,
        sample_batch_size=1,
        add_batches=False,
    )
    init_sample = {k: v[0] for k, v in trajs.items()}
    buffer_state = buffer.init(init_sample)

    # buffer state exp init
    buffer_state_exp = {k: v[None] for k, v in trajs.items()}

    buffer_state = buffer_state.replace(
        experience=buffer_state_exp,
        is_full=True,
        current_index=0,
    )
    return buffer_state

def sample_random_trajectory(
    seed: int,
    env: gym.Env,
    n_samples: int,
):
    trajs = {traj_key: [] for traj_key in TRAJ_KEYS}

    while True:
        observation, _, done, truncated = *env.reset(seed=seed), False, False
        while not (done or truncated):
            action = env.action_space.sample()
            next_observation, reward, done, truncated, _ = env.step(action)

            trajs["observations"].append(observation)
            trajs["actions"].append(action)
            trajs["rewards"].append(reward)
            trajs["dones"].append(done)
            trajs["truncated"].append(done or truncated)
            trajs["observations_next"].append(next_observation)

            observation = next_observation

            if len(trajs["observations"]) == n_samples:
                for k, v in trajs.items():
                    trajs[k] = np.stack(v)
                return trajs

        seed += 1

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
        while not (done or truncated):
            action = agent.eval_actions(observation)
            next_observation, reward, done, truncated, _ = env.step(action)

            trajs["observations"].append(observation)
            trajs["actions"].append(action)
            trajs["rewards"].append(reward)
            trajs["dones"].append(done)
            trajs["truncated"].append(done or truncated)
            trajs["observations_next"].append(next_observation)

            observation = next_observation

    stats = {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}
    if return_trajectories:
        for k, v in trajs.items():
            trajs[k] = np.stack(v)
        return stats, trajs
    return stats
