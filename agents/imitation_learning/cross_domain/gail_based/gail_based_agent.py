from typing import Optional

import gymnasium as gym
import jax
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from agents.imitation_learning.cross_domain.domain_encoder.base_domain_encoder import \
    BaseDomainEncoder
from agents.imitation_learning.in_domain.gail import GAILAgent
from utils.custom_types import DataType


class GAILBasedAgent(GAILAgent):
    domain_encoder: BaseDomainEncoder
    freeze_domain_encoder: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        encoding_dim: int,
        domain_encoder_config: DictConfig,
        freeze_domain_encoder: bool = False,
        **kwargs,
    ):
        # domain encoder init
        domain_encoder = instantiate(
            domain_encoder_config,
            seed=seed,
            encoding_dim=encoding_dim,
            _recursive_=False,
        )

        # set attrs to save
        _save_attrs = kwargs.pop(
            "_save_attrs",
            ("gail_agent", "domain_encoder")
        )

        return super().create(
            seed=seed,
            domain_encoder=domain_encoder,
            freeze_domain_encoder=freeze_domain_encoder,
            encoding_dim=encoding_dim,
            **kwargs,
        )

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_target_state(observations)

    def _preprocess_expert_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_source_state(observations)

    def update(self, batch: DataType):
        new_gail_based_agent, info, stats_info = _update_jit(
            gail_based_agent=self,
            target_expert_batch=batch,
        )
        return new_gail_based_agent, info, stats_info

    def evaluate(
        self,
        *,
        seed: int,
        env: gym.Env,
        num_episodes: int,
        convert_to_wandb_type: bool = True,
        return_trajectories: bool = False,
        return_traj_dict: bool = False,
        two_dim_data_plot_flag: bool = False,
    ):
        eval_info, trajs, traj_dict = super().evaluate(
            seed=seed,
            env=env,
            num_episodes=num_episodes,
            convert_to_wandb_type=convert_to_wandb_type,
            return_trajectories=True,
            return_traj_dict=True,
        )

        domain_encoder_eval_info = self.domain_encoder.evaluate(
            seed=seed,
            traj_dict=traj_dict,
            two_dim_data_plot_flag=two_dim_data_plot_flag,
            convert_to_wandb_type=convert_to_wandb_type,
        )
        eval_info.update(domain_encoder_eval_info)

        #
        if return_trajectories and return_traj_dict:
            return eval_info, trajs, traj_dict
        if return_traj_dict:
            return eval_info, traj_dict
        if return_trajectories:
            return eval_info, trajs
        return eval_info


def _update_jit(gail_based_agent: GAILBasedAgent, target_expert_batch: DataType):
    return _update_no_das_jit(
        gail_based_agent=gail_based_agent,
        target_expert_batch=target_expert_batch,
    )

@jax.jit
def _update_no_das_jit(gail_based_agent: GAILBasedAgent, target_expert_batch: DataType):
    # update domain encoder
    (
        new_domain_encoder,
        encoded_target_expert_batch,
        encoded_source_random_batch,
        encoded_source_expert_batch,
        info,
        stats_info,
    ) = gail_based_agent.domain_encoder.update(target_expert_batch=target_expert_batch)

    new_domain_encoder = jax.lax.cond(
        gail_based_agent.freeze_domain_encoder,
        lambda: gail_based_agent.domain_encoder,
        lambda: new_domain_encoder,
    )

    new_gail_based_agent = gail_based_agent.replace(domain_encoder=new_domain_encoder)

    # update agent and policy discriminator
    new_gail_based_agent, gail_info, gail_stats_info = new_gail_based_agent.update_with_expert_batch_given(
        target_expert_batch=target_expert_batch,
        source_expert_batch=encoded_source_expert_batch,
        policy_discriminator_target_expert_batch=encoded_target_expert_batch,
    )

    info.update(gail_info)
    stats_info.update(gail_stats_info)
    return new_gail_based_agent, info, stats_info
