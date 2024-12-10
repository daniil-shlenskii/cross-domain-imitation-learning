from copy import deepcopy
from typing import Callable

import gymnasium as gym
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import wandb
from agents.gail.gail_agent import GAILAgent
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils import convert_figure_to_array, get_buffer_state_size
from utils.types import BufferState, DataType
from utils.utils import sample_batch

from .domain_encoder.base_domain_encoder import BaseDomainEncoder
from .utils import (get_discriminators_hists,
                    get_state_and_policy_tsne_scatterplots)


class DIDAAgent(GAILAgent):
    domain_encoder: BaseDomainEncoder
    anchor_buffer_state: BufferState = struct.field(pytree_node=False)
    das: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        observation_dim: gym.Space,
        action_dim: gym.Space,
        low: np.ndarray[float],
        high: np.ndarray[float],
        #
        encoding_dim: int,
        #
        expert_batch_size: int,
        expert_buffer_state_path: str,
        #
        agent_config: DictConfig,
        policy_discriminator_config: DictConfig,
        domain_encoder_config: DictConfig,
        #
        das_config: DictConfig = None,
        #
        expert_buffer_state_preprocessing_config: DictConfig = None,
        **kwargs,
    ):
        # domain encoder init
        domain_encoder = instantiate(
            domain_encoder_config,
            seed=seed,
            learner_dim=observation_dim,
            encoding_dim=encoding_dim,
            _recursive_=False,
        )
        # DAS init
        das = None
        if das_config is not None:
            das = instantiate(das_config)

        _save_attrs = kwargs.pop(
            "_save_attrs",
            (
                "agent",
                "policy_discriminator",
                "domain_encoder",
                "das",
            )
        )

        dida_agent = super().create(
            seed=seed,
            observation_dim=encoding_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            expert_batch_size=expert_batch_size,
            expert_buffer_state_path=expert_buffer_state_path,
            agent_config=agent_config,
            policy_discriminator_config=policy_discriminator_config,
            #
            anchor_buffer_state=None,
            domain_encoder=domain_encoder,
            das=das,
            _save_attrs=_save_attrs,
            **kwargs,
        )

        # preprocess expert buffer if needed
        if expert_buffer_state_preprocessing_config is not None:
            expert_buffer_state_preprocessing = instantiate(expert_buffer_state_preprocessing_config)
            new_expert_buffer_state = expert_buffer_state_preprocessing(dida_agent.expert_buffer_state)
            dida_agent = dida_agent.replace(expert_buffer_state=new_expert_buffer_state)

        # anchor buffer init
        buffer_state_size = get_buffer_state_size(dida_agent.expert_buffer_state)
        anchor_buffer_state = deepcopy(dida_agent.expert_buffer_state)
        perm_idcs = np.random.choice(buffer_state_size)

        buffer_data = anchor_buffer_state.experience["observations_next"].dtype
        if isinstance(buffer_data, np.ndarray):
            anchor_buffer_state.experience["observations_next"][0] = \
                    anchor_buffer_state.experience["observations_next"][0, perm_idcs]
        else:
            anchor_buffer_state.experience["observations_next"] = \
                anchor_buffer_state.experience["observations_next"].at[0].set(
                    anchor_buffer_state.experience["observations_next"][0, perm_idcs]
                )

        dida_agent = dida_agent.replace(anchor_buffer_state=anchor_buffer_state)
        return dida_agent

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_learner_state(observations)

    def _preprocess_expert_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_expert_state(observations)

    def update(self, batch: DataType):
        new_dida_agent, info, stats_info = _update_jit(
            dida_agent=self,
            learner_batch=batch,
        )
        return new_dida_agent, info, stats_info

    def evaluate(
        self,
        *,
        seed: int,
        env: gym.Env,
        num_episodes: int,
        #
        visualize_state_and_policy_scatterplots: bool = False,
        visualize_state_and_policy_histograms: bool = False,
        convert_to_wandb_type: bool = True,
    ):
        eval_info, trajs = super().evaluate(seed=seed, env=env, num_episodes=num_episodes, return_trajectories=True)

        # state and policy scatterplots
        if visualize_state_and_policy_scatterplots:
            tsne_state_figure, tsne_policy_figure = get_state_and_policy_tsne_scatterplots(
                dida_agent=self,
                seed=seed,
                learner_trajs=trajs,
            )
            if convert_to_wandb_type:
                tsne_state_figure = wandb.Image(convert_figure_to_array(tsne_state_figure), caption="TSNE plot of state feautures")
                tsne_policy_figure = wandb.Image(convert_figure_to_array(tsne_policy_figure), caption="TSNE plot of policy feautures")
            eval_info["tsne_state_scatter"] = tsne_state_figure
            eval_info["tsne_policy_scatter"] = tsne_policy_figure

        # domain discriminator historgrams
        if visualize_state_and_policy_histograms:
            (
                state_learner_hist,
                state_expert_hist,
                policy_learner_hist,
                policy_expert_hist
            ) = get_discriminators_hists(
                dida_agent=self,
                learner_trajs=trajs,
)
            if convert_to_wandb_type:
                state_learner_hist = wandb.Image(convert_figure_to_array(state_learner_hist), caption="Domain Discriminator Learner logits")
                state_expert_hist = wandb.Image(convert_figure_to_array(state_expert_hist), caption="Domain Discriminator Expert logits")
                policy_learner_hist = wandb.Image(convert_figure_to_array(policy_learner_hist), caption="Policy Discriminator Learner logits")
                policy_expert_hist = wandb.Image(convert_figure_to_array(policy_expert_hist), caption="Policy Discriminator Expert logits")
            eval_info["state_learner_hist"] = state_learner_hist
            eval_info["state_expert_hist"] = state_expert_hist
            eval_info["policy_learner_hist"] = policy_learner_hist
            eval_info["policy_expert_hist"] = policy_expert_hist

        return eval_info

def _update_jit(dida_agent: DIDAAgent, learner_batch: DataType):
    # sample batches and update domain encoder
    (
        new_dida_agent,
        batch,
        expert_batch,
        anchor_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info,
    ) = _update_domain_encoder(
        dida_agent=dida_agent,
        learner_batch=learner_batch,
    )

    # prepare mixed batch for policy discriminator update
    if new_dida_agent.das is not None:
        # mix learner and anchor batches
        mixed_batch, sar_info = new_dida_agent.das.mix_batches(
            learner_batch=batch,
            anchor_batch=anchor_batch,
            learner_domain_logits=learner_domain_logits,
            expert_domain_logits=expert_domain_logits,
        )
        info.update(sar_info)
    else:
        mixed_batch = batch

    # update agent and policy discriminator
    new_dida_agent, gail_info, gail_stats_info = new_dida_agent.update_gail(
        batch=batch,
        expert_batch=expert_batch,
        policy_discriminator_learner_batch=mixed_batch,
        #
        update_agent=True,
        sample_discriminator_expert_batch=expert_batch,
    )

    info.update(gail_info)
    stats_info.update(gail_stats_info)
    return new_dida_agent, info, stats_info

def _update_domain_encoder(
    dida_agent: DIDAAgent,
    learner_batch: DataType,
):
    # sample source batches
    new_rng, expert_batch = sample_batch(dida_agent.rng, dida_agent.expert_buffer, dida_agent.expert_buffer_state)
    new_rng, anchor_batch = sample_batch(new_rng, dida_agent.expert_buffer, dida_agent.anchor_buffer_state)

    # update domain encoder
    (
        new_domain_encoder,
        learner_batch,
        expert_batch,
        anchor_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info,
    ) = dida_agent.domain_encoder.update(
        learner_batch=learner_batch,
        expert_batch=expert_batch,
        anchor_batch=anchor_batch,
    )

    # update dida agent
    new_dida_agent = dida_agent.replace(
        rng=new_rng,
        domain_encoder=new_domain_encoder
    )

    return (
        new_dida_agent,
        learner_batch,
        expert_batch,
        anchor_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info,
    )
