from copy import deepcopy
from typing import Callable

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from flax import struct
from typing_extensions import override

import wandb
from agents.base_agent import Agent
from utils import (convert_figure_to_array, get_buffer_state_size,
                   instantiate_jitted_fbx_buffer, load_pickle)
from utils.types import Buffer, BufferState

from .utils import get_state_and_policy_tsne_scatterplots


class ImitationAgent(Agent):
    expert_buffer: Buffer = struct.field(pytree_node=False)
    expert_buffer_state: BufferState = struct.field(pytree_node=False)

    @override
    def _preprocess_expert_observations(self, observations: np.ndarray) -> np.ndarray:
        return observations

    def evaluate(
        self,
        *,
        seed: int,
        env: gym.Env,
        num_episodes: int,
        #
        convert_to_wandb_type: bool = True,
        #
        return_trajectories: bool = False,
    ):
        eval_info, trajs = super().evaluate(seed=seed, env=env, num_episodes=num_episodes, return_trajectories=True)

        # visualize state and policy encodings
        tsne_state_figure, tsne_policy_figure = get_state_and_policy_tsne_scatterplots(
            imitation_agent=self, seed=seed, learner_trajs=trajs,
        )
        if convert_to_wandb_type:
            tsne_state_figure = wandb.Image(convert_figure_to_array(tsne_state_figure), caption="TSNE plot of state feautures")
            tsne_policy_figure = wandb.Image(convert_figure_to_array(tsne_policy_figure), caption="TSNE plot of policy feautures")
        eval_info["tsne_state_scatter"] = tsne_state_figure
        eval_info["tsne_policy_scatter"] = tsne_policy_figure

        #
        if return_trajectories:
            return eval_info, trajs
        return eval_info
