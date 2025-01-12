import gymnasium as gym
import numpy as np
from flax import struct
from typing_extensions import override

import wandb
from agents.base_agent import Agent
from utils import convert_figure_to_array
from utils.custom_types import Buffer, BufferState

from .utils import (get_state_and_policy_tsne_scatterplots, get_state_pairs,
                    get_trajectory_from_buffer, get_trajectory_from_dict,
                    get_trajs_tsne_scatterplot)


class ImitationAgent(Agent):
    buffer: Buffer = struct.field(pytree_node=False)
    source_expert_buffer_state: BufferState = struct.field(pytree_node=False)

    @override
    def _preprocess_expert_observations(self, observations: np.ndarray) -> np.ndarray:
        return observations

    def evaluate(
        self,
        *,
        seed: int,
        env: gym.Env,
        num_episodes: int,
        return_trajectories: bool = False,
        #
        return_traj_dict: bool = False,
        convert_to_wandb_type: bool = True,
    ):
        eval_info, trajs = super().evaluate(seed=seed, env=env, num_episodes=num_episodes, return_trajectories=True)

        # get learner and expert trajectories for visualization
        target_expert_traj = get_trajectory_from_dict(trajs)
        source_expert_traj = get_trajectory_from_buffer(self.source_expert_buffer_state)

        # preprocess trajectories
        for k in ["observations", "observations_next"]:
            target_expert_traj[k] = self._preprocess_observations(target_expert_traj[k])
            source_expert_traj[k] = self._preprocess_expert_observations(source_expert_traj[k])

        # traj_dict
        traj_dict = {
            "states": {
                "TE": target_expert_traj["observations"],
                "SE": source_expert_traj["observations"],
            },
            "state_pairs": {
                "TE": get_state_pairs(target_expert_traj),
                "SE": get_state_pairs(source_expert_traj),
            },
        }

        # get states tsne scatterplots
        state_tsne_scatterplot = get_trajs_tsne_scatterplot(
            traj_dict=traj_dict["states"],
            keys_to_use=["TE", "SE"],
            seed=seed,
        )
        if convert_to_wandb_type:
            state_tsne_scatterplot = wandb.Image(convert_figure_to_array(state_tsne_scatterplot), caption="TSNE plot of state feautures")
        eval_info["tsne_state_scatter"] = state_tsne_scatterplot

        #
        to_return = [eval_info]
        if return_trajectories:
            to_return.append(trajs)
        if return_traj_dict:
            to_return.append(traj_dict)
        return to_return
