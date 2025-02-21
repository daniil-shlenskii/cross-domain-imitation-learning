from typing import Optional

import gymnasium as gym
import jax
import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import wandb
from agents.imitation_learning.base_imitation_agent import ImitationAgent
from agents.imitation_learning.in_domain.gail.utils import \
    get_trajs_discriminator_logits_and_accuracy
from agents.imitation_learning.utils import prepare_buffer
from utils import convert_figure_to_array, sample_batch_jit
from utils.custom_types import DataType

from .gail_discriminator import GAILDiscriminator


class GAILAgent(ImitationAgent):
    policy_discriminator: GAILDiscriminator

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        observation_dim: int,
        action_dim: int,
        low: np.ndarray,
        high: np.ndarray,
        #
        agent_config: DictConfig,
        policy_discriminator_config: DictConfig,
        #
        source_expert_buffer_state_path: str,
        batch_size: Optional[int] = 1,
        sourse_buffer_processor_config: Optional[DictConfig] = None,
        encoding_dim = None,
        **kwargs,
    ):
        # agent and policy_discriminator init
        agent = instantiate(
            agent_config,
            seed=seed,
            observation_dim=observation_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            _recursive_=False,
        )

        encoding_dim = encoding_dim if encoding_dim is not None else observation_dim
        policy_discriminator = instantiate(
            policy_discriminator_config,
            seed=seed,
            input_dim=encoding_dim * 2,
            _recursive_=False,
        )

        # expert buffer init
        buffer, source_expert_buffer_state = prepare_buffer(
            buffer_state_path=source_expert_buffer_state_path,
            batch_size=batch_size,
            sourse_buffer_processor_config=sourse_buffer_processor_config,
        )

        # set attrs to save
        _save_attrs = kwargs.pop(
            "_save_attrs",
            ("agent", "policy_discriminator")
        )

        return cls(
            rng=jax.random.key(seed),
            buffer=buffer,
            source_expert_buffer_state=source_expert_buffer_state,
            agent=agent,
            policy_discriminator=policy_discriminator,
            _save_attrs = _save_attrs,
            **kwargs,
        )

    @property
    def actor(self):
        return self.agent.actor

    def update(self, batch: DataType):
        new_gail_agent, info, stats_info = _update_jit(
            gail_agent=self, target_expert_batch=batch
        )
        return new_gail_agent, info, stats_info

    def update_with_expert_batch_given(
        self,
        *,
        target_expert_batch: DataType,
        source_expert_batch: DataType,
        policy_discriminator_target_expert_batch: DataType,
    ):
        return _update_with_expert_batch_given_jit(
            gail_agent=self,
            target_expert_batch=target_expert_batch,
            source_expert_batch=source_expert_batch,
            policy_discriminator_target_expert_batch=policy_discriminator_target_expert_batch,
        )

    def evaluate(
        self,
        *,
        seed: int,
        env: gym.Env,
        num_episodes: int,
        #
        return_trajectories: bool = False,
        #
        return_traj_dict: bool = False,
        convert_to_wandb_type: bool = True,
    ):
        eval_info, trajs, traj_dict = super().evaluate(seed=seed, env=env, num_episodes=num_episodes, return_trajectories=True, return_traj_dict=True)

        # gail discriminator eval info
        policy_discriminator_info =self.policy_discriminator.evaluate(traj_dict, convert_to_wandb_type=convert_to_wandb_type)
        eval_info.update(**policy_discriminator_info)

        #
        if return_trajectories and return_traj_dict:
            return eval_info, trajs, traj_dict
        if return_traj_dict:
            return eval_info, traj_dict
        if return_trajectories:
            return eval_info, trajs
        return eval_info

@jax.jit
def _update_jit(*,gail_agent: GAILAgent, target_expert_batch: DataType):
    # sample expert batch
    new_rng, source_expert_batch = sample_batch_jit(
        gail_agent.rng, gail_agent.buffer, gail_agent.source_expert_buffer_state
    )
    new_gail_agent = gail_agent.replace(rng=new_rng)

    # update agent and policy policy_discriminator
    new_gail_agent, info, stats_info = _update_with_expert_batch_given_jit(
        gail_agent=new_gail_agent,
        target_expert_batch=target_expert_batch,
        source_expert_batch=source_expert_batch,
        policy_discriminator_target_expert_batch=target_expert_batch,
    )
    return new_gail_agent, info, stats_info

@jax.jit
def _update_with_expert_batch_given_jit(
    *,
    gail_agent: GAILAgent,
    target_expert_batch: DataType,
    source_expert_batch: DataType,
    policy_discriminator_target_expert_batch: DataType,
):
    # update policy discriminator
    new_disc, disc_info, disc_stats_info = gail_agent.policy_discriminator.update(
        target_expert_batch=policy_discriminator_target_expert_batch,
        source_expert_batch=source_expert_batch,
    )

    # update agent
    target_expert_batch["rewards"] = new_disc.get_rewards(policy_discriminator_target_expert_batch)
    new_agent, agent_info, agent_stats_info = gail_agent.agent.update(target_expert_batch)

    # update gail agent
    new_gail_agent = gail_agent.replace(
        agent=new_agent,
        policy_discriminator=new_disc,
    )

    info = {**disc_info, **agent_info}
    stats_info = {**disc_stats_info, **agent_stats_info}
    return new_gail_agent, info, stats_info
