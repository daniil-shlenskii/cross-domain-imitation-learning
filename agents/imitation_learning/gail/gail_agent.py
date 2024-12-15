import gymnasium as gym
import jax
import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from agents.imitation_learning.base_imitation_agent import ImitationAgent
from utils import sample_batch
from utils.types import DataType

from .gail_discriminator import GAILDiscriminator


class GAILAgent(ImitationAgent):
    agent: Agent
    policy_discriminator: GAILDiscriminator

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
        expert_batch_size: int,
        expert_buffer_state_path: str,
        #
        agent_config: DictConfig,
        #
        policy_discriminator_config: DictConfig,
        #
        expert_buffer_state_processor_config: DictConfig = None,
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

        policy_discriminator = instantiate(
            policy_discriminator_config,
            seed=seed,
            input_dim=observation_dim * 2,
            _recursive_=False,
        )

        # expert buffer init
        expert_buffer_state_processor = None
        if expert_buffer_state_processor_config is not None:
            expert_buffer_state_processor = instantiate(expert_buffer_state_processor)

        expert_buffer, expert_buffer_state = cls._prepare_expert_buffer(
            expert_buffer_state_path=expert_buffer_state_path,
            expert_batch_size=expert_batch_size,
            expert_buffer_state_processor=expert_buffer_state_processor,
        )

        # set attrs to save
        _save_attrs = kwargs.pop(
            "_save_attrs",
            ("agent", "policy_discriminator")
        )

        return cls(
            rng=jax.random.key(seed),
            expert_buffer=expert_buffer,
            expert_buffer_state=expert_buffer_state,
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
            gail_agent=self, learner_batch=batch
        )
        return new_gail_agent, info, stats_info

    def update_gail(
        self,
        *,
        learner_batch: DataType,
        expert_batch: DataType,
        policy_discriminator_learner_batch: DataType,
    ):
        return _update_gail_jit(
            gail_agent=self,
            learner_batch=learner_batch,
            expert_batch=expert_batch,
            policy_discriminator_learner_batch=policy_discriminator_learner_batch,
        )

@jax.jit
def _update_jit(gail_agent: GAILAgent, learner_batch: DataType):
    # sample expert batch
    new_rng, expert_batch = sample_batch(
        gail_agent.rng, gail_agent.expert_buffer, gail_agent.expert_buffer_state
    )
    new_gail_agent = gail_agent.replace(rng=new_rng)

    # update agent and policy policy_discriminator
    new_gail_agent, info, stats_info = new_gail_agent.update_gail(
        learner_batch=learner_batch,
        expert_batch=expert_batch,
        policy_discriminator_learner_batch=learner_batch,
    )
    return new_gail_agent, info, stats_info

@jax.jit
def _update_gail_jit(
    gail_agent: GAILAgent,
    learner_batch: DataType,
    expert_batch: DataType,
    policy_discriminator_learner_batch: DataType,
):
    # update policy discriminator
    new_disc, disc_info, disc_stats_info = gail_agent.policy_discriminator.update(
        learner_batch=policy_discriminator_learner_batch,
        expert_batch=expert_batch,
    )

    # update agent
    learner_batch["rewards"] = new_disc.get_rewards(learner_batch)
    new_agent, agent_info, agent_stats_info = gail_agent.agent.update(learner_batch)

    # update gail agent
    new_gail_agent = gail_agent.replace(
        agent=new_agent,
        policy_discriminator=new_disc,
    )

    info = {**disc_info, **agent_info}
    stats_info = {**disc_stats_info, **agent_stats_info}
    return new_gail_agent, info, stats_info
