import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import wandb
from agents.imitation_learning.gail import GAILAgent
from utils import convert_figure_to_array
from utils.types import DataType

from .domain_encoder.base_domain_encoder import BaseDomainEncoder
from .utils import get_discriminators_logits_plots


class DIDAAgent(GAILAgent):
    domain_encoder: BaseDomainEncoder
    freeze_domain_encoder: bool = struct.field(pytree_node=False)
    das: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        #
        encoding_dim: int,
        #
        expert_batch_size: int,
        expert_buffer_state_path: str,
        #
        domain_encoder_config: DictConfig,
        freeze_domain_encoder: bool = False,
        #
        das_config: DictConfig = None,
        #
        expert_buffer_state_processor_config: DictConfig = None,
        **kwargs,
    ):
        # domain encoder init
        domain_encoder = instantiate(
            domain_encoder_config,
            seed=seed,
            encoding_dim=encoding_dim,
            source_buffer_state_processor_config=expert_buffer_state_processor_config,
            _recursive_=False,
        )

        # DAS init
        das = None
        if das_config is not None:
            das = instantiate(das_config)

        # set attrs to save
        _save_attrs = kwargs.pop(
            "_save_attrs",
            ("agent", "policy_discriminator", "domain_encoder")
        )

        kwargs["observation_dim"] = encoding_dim
        return super().create(
            seed=seed,
            expert_batch_size=expert_batch_size,
            expert_buffer_state_path=expert_buffer_state_path,
            expert_buffer_state_processor_config=expert_buffer_state_processor_config,
            #
            domain_encoder=domain_encoder,
            freeze_domain_encoder=freeze_domain_encoder,
            das=das,
            _save_attrs=_save_attrs,
            **kwargs,
        )

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_target_state(observations)

    def _preprocess_expert_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_source_state(observations)

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
        convert_to_wandb_type: bool = True,
        return_trajectories: bool = False
    ):
        eval_info, trajs = super().evaluate(
            seed=seed,
            env=env,
            num_episodes=num_episodes,
            convert_to_wandb_type=convert_to_wandb_type,
            return_trajectories=True,
        )

        domain_encoder_eval_info = self.domain_encoder.evaluate(seed=seed)
        eval_info.update(domain_encoder_eval_info)

        target_state_plot, source_state_plot, target_policy_plot, source_policy_plot = \
            get_discriminators_logits_plots(dida_agent=self, learner_trajs=trajs)
        if convert_to_wandb_type:
            target_state_plot = wandb.Image(convert_figure_to_array(target_state_plot), caption="(Discriminators) State Discriminator Learner logits")
            source_state_plot = wandb.Image(convert_figure_to_array(source_state_plot), caption="(Discriminators) State Discriminator Expert logits")
            target_policy_plot = wandb.Image(convert_figure_to_array(target_policy_plot), caption="(Discriminators) Policy Discriminator Learner logits")
            source_policy_plot = wandb.Image(convert_figure_to_array(source_policy_plot), caption="(Discriminators) Policy Discriminator Expert logits")
        eval_info["target_state_plot"] = target_state_plot
        eval_info["source_state_plot"] = source_state_plot
        eval_info["target_policy_plot"] = target_policy_plot
        eval_info["source_policy_plot"] = source_policy_plot

        if return_trajectories:
            return eval_info, trajs
        return eval_info

def _update_jit(dida_agent: DIDAAgent, learner_batch: DataType):
    return _update_no_das_jit(
        dida_agent=dida_agent,
        learner_batch=learner_batch,
    )

@jax.jit
def _update_no_das_jit(dida_agent: DIDAAgent, learner_batch: DataType):
    # update domain encoder
    (
        new_domain_encoder,
        target_expert_batch,
        source_random_batch,
        source_expert_batch,
        info,
        stats_info,
    ) = dida_agent.domain_encoder.update(target_expert_batch=learner_batch)

    new_domain_encoder = jax.lax.cond(
        dida_agent.freeze_domain_encoder,
        lambda: dida_agent.domain_encoder,
        lambda: new_domain_encoder,
    )

    new_dida_agent = dida_agent.replace(domain_encoder=new_domain_encoder)

    # update agent and policy discriminator
    policy_discriminator_learner_batch = {
        k: jnp.concatenate([
            target_expert_batch[k],
            source_random_batch[k],
        ])
        for k in target_expert_batch
    }

    new_dida_agent, gail_info, gail_stats_info = new_dida_agent.update_gail(
        learner_batch=target_expert_batch,
        expert_batch=source_expert_batch,
        policy_discriminator_learner_batch=policy_discriminator_learner_batch,
    )

    info.update(gail_info)
    stats_info.update(gail_stats_info)
    return new_dida_agent, info, stats_info
