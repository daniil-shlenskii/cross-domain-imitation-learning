from copy import deepcopy
from typing import Callable

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import wandb
from flax import struct
from typing_extensions import override

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

    @classmethod
    def _prepare_expert_buffer(
        cls,
        expert_buffer_state_path: str,
        expert_batch_size: int,
        expert_buffer_state_processor: Callable = None,
    ):
        # load buffer state
        _expert_buffer_state = load_pickle(expert_buffer_state_path)

        # buffer init
        buffer_state_size = get_buffer_state_size(_expert_buffer_state)
        expert_buffer = instantiate_jitted_fbx_buffer({
            "_target_": "flashbax.make_item_buffer",
            "sample_batch_size": expert_batch_size,
            "min_length": 1,
            "max_length": buffer_state_size,
            "add_batches": False,
        })

        # remove dummy experience samples (flashbax specific stuff)
        expert_buffer_state_exp = _expert_buffer_state.experience
        buffer_state_init_sample = {k: v[0, 0] for k, v in expert_buffer_state_exp.items()}
        expert_buffer_state = expert_buffer.init(buffer_state_init_sample)

        new_expert_buffer_state_exp = {}
        for k, v in expert_buffer_state_exp.items():
            new_expert_buffer_state_exp[k] = jnp.asarray(v[0, :buffer_state_size][None])

        expert_buffer_state = expert_buffer_state.replace(
            experience=new_expert_buffer_state_exp,
            current_index=0,
            is_full=True,
        )

        # process expert buffer state
        if expert_buffer_state_processor is not None:
            expert_buffer_state = expert_buffer_state_processor(expert_buffer_state)

        return expert_buffer, expert_buffer_state

    @classmethod
    def _get_source_random_buffer_state(cls, expert_buffer_state: BufferState, seed: int):
        buffer_state_size = get_buffer_state_size(expert_buffer_state)
        anchor_buffer_state = deepcopy(expert_buffer_state)

        np.random.seed(seed)
        obs_perm_idcs = np.random.choice(buffer_state_size)
        obs_next_perm_idcs = np.random.choice(buffer_state_size)

        anchor_buffer_state.experience["observations"] = \
            anchor_buffer_state.experience["observations"].at[0].set(
                anchor_buffer_state.experience["observations"][0, obs_perm_idcs]
            )

        anchor_buffer_state.experience["observations_next"] = \
            anchor_buffer_state.experience["observations_next"].at[0].set(
                anchor_buffer_state.experience["observations_next"][0, obs_next_perm_idcs]
            )

        return anchor_buffer_state

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
