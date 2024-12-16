import gymnasium as gym
import jax
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.imitation_learning.gail import GAILAgent
from utils import sample_batch
from utils.types import BufferState, DataType

from .domain_encoder.base_domain_encoder import BaseDomainEncoder


class DIDAAgent(GAILAgent):
    domain_encoder: BaseDomainEncoder
    anchor_buffer_state: BufferState = struct.field(pytree_node=False)
    das: float = struct.field(pytree_node=False)
    n_iters_encoder_pretrain: int = struct.field(pytree_node=False)

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
        n_iters_encoder_pretrain: int = 0,
        #
        das_config: DictConfig = None,
        #
        expert_buffer_state_processor_config: DictConfig = None,
        **kwargs,
    ):
        # DAS init
        das = None
        if das_config is not None:
            das = instantiate(das_config)

        # gail agent init
        gail_agent = super().create(
            seed=seed,
            observation_dim=encoding_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            expert_batch_size=expert_batch_size,
            expert_buffer_state_path=expert_buffer_state_path,
            agent_config=agent_config,
            policy_discriminator_config=policy_discriminator_config,
            expert_buffer_state_processor_config=expert_buffer_state_processor_config,
            #
            domain_encoder=None,
            anchor_buffer_state=None,
            das=das,
            n_iters_encoder_pretrain=n_iters_encoder_pretrain,
            **kwargs,
        )

        # anchor buffer init
        anchor_buffer_state = cls._get_source_random_buffer_state(
            expert_buffer_state=gail_agent.expert_buffer_state,
            seed=seed,
        )

        # domain encoder init
        expert_observation_dim = gail_agent.expert_buffer_state.experience["observations"].shape[-1]
        domain_encoder = instantiate(
            domain_encoder_config,
            seed=seed,
            learner_dim=observation_dim,
            expert_dim=expert_observation_dim,
            encoding_dim=encoding_dim,
            _recursive_=False,
        )

        # _save_attrs update
        _save_attrs = gail_agent._save_attrs + ("domain_encoder",)

        # dida agent init
        dida_agent = gail_agent.replace(
            anchor_buffer_state=anchor_buffer_state,
            domain_encoder=domain_encoder,
            _save_attrs=_save_attrs,
        )
        return dida_agent

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_learner_state(observations)

    def _preprocess_expert_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_expert_state(observations)

    def update(self, batch: DataType):
        if self.domain_encoder.learner_encoder.state.step < self.n_iters_encoder_pretrain:
            # pretrain domain encoder
            new_dida_agent, _, _, _, info, stats_info = _update_domain_encoder_jit(
                dida_agent=self,
                learner_batch=batch,
            )
        else:
            # main agent update
            new_dida_agent, info, stats_info = _update_jit(
                dida_agent=self,
                learner_batch=batch,
            )
        return new_dida_agent, info, stats_info

def _update_jit(dida_agent: DIDAAgent, learner_batch: DataType):
    if dida_agent.das is None:
        update_func = _update_no_das_jit
    else:
        update_func = _update_with_das_jit
    return update_func(
        dida_agent=dida_agent,
        learner_batch=learner_batch,
    )

def _update_with_das_jit(dida_agent: DIDAAgent, learner_batch: DataType):
    # sample batches and update domain encoder
    (
        new_dida_agent,
        learner_batch,
        expert_batch,
        intermediates,
        info,
        stats_info,
    ) = _update_domain_encoder_jit(
        dida_agent=dida_agent,
        learner_batch=learner_batch,
    )

    # das: mix learner and anchor batches
    mixed_batch, sar_info = new_dida_agent.das.mix_batches(
        learner_batch=learner_batch,
        anchor_batch=intermediates["anchor_batch"],
        learner_domain_logits=intermediates["learner_domain_logits"],
        expert_domain_logits=intermediates["expert_domain_logits"],
    )
    info.update(sar_info)

    # update agent and policy discriminator
    new_dida_agent, gail_info, gail_stats_info = new_dida_agent.update_gail(
        learner_batch=learner_batch,
        expert_batch=expert_batch,
        policy_discriminator_learner_batch=mixed_batch,
    )
    info.update(gail_info)
    stats_info.update(gail_stats_info)

    return new_dida_agent, info, stats_info

@jax.jit
def _update_no_das_jit(dida_agent: DIDAAgent, learner_batch: DataType):
    # sample batches and update domain encoder
    (
        new_dida_agent,
        learner_batch,
        expert_batch,
        _,
        info,
        stats_info,
    ) = _update_domain_encoder_jit(
        dida_agent=dida_agent,
        learner_batch=learner_batch,
    )

    # update agent and policy discriminator
    new_dida_agent, gail_info, gail_stats_info = new_dida_agent.update_gail(
        learner_batch=learner_batch,
        expert_batch=expert_batch,
        policy_discriminator_learner_batch=learner_batch,
    )
    info.update(gail_info)
    stats_info.update(gail_stats_info)

    return new_dida_agent, info, stats_info

@jax.jit
def _update_domain_encoder_jit(
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
        intermediates,
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
        intermediates,
        info,
        stats_info,
    )
