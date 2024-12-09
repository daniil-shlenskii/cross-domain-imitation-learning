from copy import deepcopy
from typing import Callable

import gymnasium as gym
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import wandb
from agents.gail.gail_agent import GAILAgent
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils import (apply_model_jit, convert_figure_to_array,
                   get_buffer_state_size)
from utils.types import BufferState, DataType

from .das import DomainAdversarialSampling, _prepare_anchor_batch_jit
from .domain_loss_scale_fns import ConstantDomainLossScale
from .update_steps import (_update_domain_discriminator_only_jit,
                           _update_encoders_and_domain_discriminator_jit)
from .utils import (get_discriminators_hists,
                    get_state_and_policy_tsne_scatterplots)


class DIDAAgent(GAILAgent):
    learner_encoder: Generator
    domain_discriminator: Discriminator
    anchor_buffer_state: BufferState = struct.field(pytree_node=False)
    das: float = struct.field(pytree_node=False)
    n_domain_discriminator_updates: int = struct.field(pytree_node=False)
    domain_loss_scale_fn: Callable = struct.field(pytree_node=False)

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
        encoder_dim: int,
        #
        expert_batch_size: int,
        expert_buffer_state_path: str,
        #
        agent_config: DictConfig,
        learner_encoder_config: DictConfig,
        policy_discriminator_config: DictConfig,
        domain_discriminator_config: DictConfig,
        #
        use_das: bool = True,
        sar_p: float = 0.5,
        p_acc_ema: float = 0.9,
        p_acc_ema_decay: float = 0.999,
        #
        n_policy_discriminator_updates: int = 1,
        n_sample_discriminator_updates: int = 1,
        n_domain_discriminator_updates: int = 1,
        domain_loss_scale_fn_config: DictConfig = None,
        #
        expert_buffer_state_preprocessing_config: DictConfig = None,
        **kwargs,
    ):
        # encoders init
        policy_loss = instantiate(policy_discriminator_config["loss_config"])
        domain_loss = instantiate(domain_discriminator_config["loss_config"])
        learner_encoder_config = OmegaConf.to_container(learner_encoder_config)
        learner_encoder_config["loss_config"]["policy_loss"] = policy_loss
        learner_encoder_config["loss_config"]["domain_loss"] = domain_loss

        learner_encoder = instantiate(
            learner_encoder_config,
            seed=seed,
            input_dim=observation_dim,
            output_dim=encoder_dim,
            info_key="encoder",
            _recursive_=False,
        )

        # domain discriminators init
        domain_discriminator = instantiate(
            domain_discriminator_config,
            seed=seed,
            input_dim=encoder_dim ,
            info_key="domain_discriminator",
            _recursive_=False,
        )

        # DAS init
        das = None
        if use_das:
            das = DomainAdversarialSampling(
                sar_p=sar_p,
                p_acc_ema=p_acc_ema,
                p_acc_ema_decay=p_acc_ema_decay,
            )

        # domain loss updater init
        if domain_loss_scale_fn_config is None:
            domain_loss_scale_fn = ConstantDomainLossScale(domain_loss_scale=1.0)
        else:
            domain_loss_scale_fn = instantiate(domain_loss_scale_fn_config)

        _save_attrs = kwargs.pop(
            "_save_attrs",
            (
                "agent",
                "learner_encoder",
                "policy_discriminator",
                "domain_discriminator",
                "das",
            )
        )

        dida_agent = super().create(
            seed=seed,
            observation_dim=encoder_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            expert_batch_size=expert_batch_size,
            expert_buffer_state_path=expert_buffer_state_path,
            agent_config=agent_config,
            policy_discriminator_config=policy_discriminator_config,
            n_policy_discriminator_updates=n_policy_discriminator_updates,
            #
            anchor_buffer_state=None,
            learner_encoder=learner_encoder,
            das=das,
            domain_discriminator=domain_discriminator,
            n_domain_discriminator_updates=n_domain_discriminator_updates,
            domain_loss_scale_fn=domain_loss_scale_fn,
            _save_attrs=_save_attrs,
            **kwargs,
        )

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
        return apply_model_jit(self.learner_encoder, observations)

    def _preprocess_expert_observations(self, observations: np.ndarray) -> np.ndarray:
        return apply_model_jit(self.expert_encoder, observations)

    def __getattr__(self, item: str):
        if item == "expert_encoder":
            return self.learner_encoder
        return super().__getattr__(item)

    def update(self, batch: DataType):
        update_domain_discriminator_only = bool(
            (self.domain_discriminator.state.step + 1) % self.n_domain_discriminator_updates != 0
        )
        update_sample_discriminator_only = bool(
            (self.sample_discriminator.state.step + 1) % self.n_sample_discriminator_updates != 0
        )
        update_dida_agent = not (
            update_domain_discriminator_only or
            update_sample_discriminator_only
        )
        if not update_dida_agent:
            new_dida_agent = self
            info, stats_info = {}, {}
            if update_domain_discriminator_only:
                new_dida_agent, domain_info, domain_stats_info = _update_domain_discriminator_only_jit(
                    dida_agent=new_dida_agent,
                    batch=batch
                )
                info.update(domain_info)
                stats_info.update(domain_stats_info)

            if update_domain_discriminator_only:
                new_dida_agent, sample_info, sample_stats_info = new_dida_agent.update_sample_discriminator(
                    batch=batch,
                    expert_encoder=new_dida_agent.expert_encoder,
                )
                info.update(sample_info)
                stats_info.update(sample_stats_info)

            return new_dida_agent, info, stats_info

        update_agent = bool(
                (self.policy_discriminator.state.step + 1) % self.n_policy_discriminator_updates == 0
        )
        new_dida_agent, info, stats_info = _update_jit(
            dida_agent=self,
            batch=batch,
            update_agent=update_agent
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

    def _update_encoders_and_domain_discrimiantor(
        self, batch: DataType, expert_batch: DataType, domain_loss_scale: float
    ):
        # update encoders
        new_encoder, encoder_info, encoder_stats_info = self.learner_encoder.update(
            batch=batch,
            expert_batch=expert_batch,
            policy_discriminator=self.policy_discriminator,
            domain_discriminator=self.domain_discriminator,
            domain_loss_scale=domain_loss_scale,
        )
        batch = encoder_info.pop("learner_encoded_batch")
        expert_batch = encoder_info.pop("expert_encoded_batch")

        # update domain discriminator
        new_domain_disc, domain_disc_info, domain_disc_stats_info = self.domain_discriminator.update(
            real_batch=expert_batch["observations"],
            fake_batch=batch["observations"],
            return_logits=True,
        )
        expert_domain_logits = domain_disc_info.pop("real_logits")
        learner_domain_logits = domain_disc_info.pop("fake_logits")

        # update dida agent
        new_dida_agent = self.replace(
            learner_encoder=new_encoder,
            domain_discriminator=new_domain_disc,
        )
        info = {**encoder_info, **domain_disc_info}
        stats_info = {**encoder_stats_info, **domain_disc_stats_info}

        return (
            new_dida_agent,
            batch,
            expert_batch,
            learner_domain_logits,
            expert_domain_logits,
            info,
            stats_info,
        )

def _update_jit(dida_agent: DIDAAgent, batch: DataType, update_agent: bool):
    # update encoders and domain discriminator
    (
        new_dida_agent,
        batch,
        expert_batch,
        sample_discr_expert_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info,
    ) = _update_encoders_and_domain_discriminator_jit(
        dida_agent=dida_agent,
        batch=deepcopy(batch),
    )

    # prepare mixed batch for policy discriminator update
    if new_dida_agent.das is not None:
        # prepare anchor batch
        anchor_batch = _prepare_anchor_batch_jit(dida_agent=dida_agent)

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
        sample_discriminator_expert_batch=sample_discr_expert_batch,
        update_agent=update_agent,
        expert_encoder=new_dida_agent.expert_encoder,
    )

    info.update(gail_info)
    stats_info.update(gail_stats_info)
    return new_dida_agent, info, stats_info
