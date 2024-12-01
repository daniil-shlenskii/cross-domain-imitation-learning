from typing import Dict

import numpy as np


def get_reward_stats(
    gail_agent: "Agent",
    trajectories: Dict[str, np.ndarray]
):
    state_pairs = np.concatenate([
        trajectories["observations"],
        trajectories["observations_next"]
    ], axis=1)
    rewards = gail_agent.discriminator.get_rewards(state_pairs)
    return {
        "rewards/mean": np.mean(rewards),
        "rewards/std": np.std(rewards),
    }
