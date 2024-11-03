import jax
import jax.numpy as jnp

import gymnasium as gym

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from nn.train_state import TrainState
from utils.types import DataType, PRNGKey, Buffer
from utils.utils import load_pickle

from gail.gail_discriminator import GAILDiscriminator


class GAILAgent(Agent):
    @classmethod
    def create(
        cls,
        *
        observations_space: gym.Space,
        action_space: gym.Space,
        #
        expert_buffer_path: str,
        #
        agent_config: DictConfig,
        discriminator_config: DictConfig
    ):
        ...
        # agent and discriminator init
        agent = instantiate(agent_config, observations_space=observations_space, action_space=action_space)
        discriminator = instantiate(discriminator_config)

        # expert buffer init
        expert_buffer = load_pickle(expert_buffer_path)

        return cls(
            expert_buffer=expert_buffer,
            agent=agent,
            discriminator=discriminator,
        )

    def __init__(
        self,
        *
        expert_buffer: Buffer,
        agent: DictConfig,
        discriminator: DictConfig,
    ):
        self.expert_buffer = expert_buffer
        self.agent = agent
        self.discriminator = discriminator

    @property
    def actor(self):
        return self.agent.actor

    def update(self, batch: DataType):
        learner_batch = jnp.concatenate([batch["observations"], batch["observations_next"]], axis=-1)
        
        batch_size = learner_batch.shape[0]
        expert_batch = self.expert_buffer.sample(batch_size)
        expert_batch = jnp.concatenate([expert_batch["observations"], expert_batch["observations_next"]], axis=-1)

        (
            self.agent,
            self.discriminator,
            info,
            stats_info
        ) = _update_jit(
            learner_batch=learner_batch,
            expert_batch=expert_batch,
            agent=self.agent,
            disc=self.discriminator,
        )

        return info, stats_info

@jax.jit
def _update_jit(
    *,
    learner_batch: jnp.ndarray,
    expert_batch: jnp.ndarray,
    #
    agent: Agent,
    disc: GAILDiscriminator,
):
    learner_batch["rewards"] = disc.get_rewards(learner_batch)
    new_agent, agent_info, agent_stats_info = agent.update(learner_batch)
    new_disc, disc_info, disc_stats_info = disc.update(expert_batch)

    info = {**agent_info, **disc_info}
    stats_info = {**agent_stats_info, **disc_stats_info}
    return (
        new_agent,
        new_disc,
        info,
        stats_info,
    )
