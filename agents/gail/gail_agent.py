from typing import Tuple

import flashbax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from agents.gail.gail_discriminator import GAILDiscriminator
from gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.types import Buffer, BufferState, DataType
from utils.utils import load_pickle


class GAILAgent(Agent):
    _save_attrs: Tuple[str] = (
        "agent",
        "discriminator"
    )

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
        discriminator_config: DictConfig,
        #
        **kwargs,
    ):
        # agent and discriminator init
        agent = instantiate(
            agent_config,
            seed=seed,
            observation_dim=observation_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            _recursive_=False,
        )
        
        obs = np.ones(observation_dim)
        discriminator_input_sample = jnp.concatenate([obs, obs], axis=-1)
        discriminator = instantiate(
            discriminator_config,
            seed=seed,
            input_sample=discriminator_input_sample,
            _recursive_=False,
        )

        # expert buffer init
        expert_buffer_state = load_pickle(expert_buffer_state_path)
        expert_buffer = flashbax.make_item_buffer(
            sample_batch_size=expert_batch_size,
            min_length=expert_buffer_state.current_index,
            max_length=expert_buffer_state.current_index,
            add_batches=False,
        )

        return cls(
            seed=seed,
            expert_buffer=expert_buffer,
            expert_buffer_state=expert_buffer_state,
            agent=agent,
            discriminator=discriminator,
            **kwargs,
        )

    def __init__(
        self,
        *,
        seed: int,
        expert_buffer: Buffer,
        expert_buffer_state: BufferState,
        agent: Agent,
        discriminator: Discriminator,
    ):
        self.rng = jax.random.key(seed=seed)
        self.expert_buffer = expert_buffer
        self.expert_buffer_state = expert_buffer_state
        self.agent = agent
        self.discriminator = discriminator

    @property
    def actor(self):
        return self.agent.actor

    def update(self, batch: DataType):
        self.rng, key = jax.random.split(self.rng)

        # process batch
        learner_batch = jnp.concatenate([batch["observations"], batch["observations_next"]], axis=-1)
        expert_batch = self.expert_buffer.sample(self.expert_buffer_state, key).experience

        expert_batch = jnp.concatenate([expert_batch["observations"], expert_batch["observations_next"]], axis=-1)

        # update agent
        batch["reward"] = self.discriminator.get_rewards(learner_batch)
        agent_info, agent_stats_info = self.agent.update(batch)

        # update discriminator
        self.discriminator, disc_info, disc_stats_info = _update_discriminator_jit(
            expert_batch=expert_batch,
            learner_batch=learner_batch,
            disc=self.discriminator,
        )
        info = {**agent_info, **disc_info}
        stats_info = {**agent_stats_info, **disc_stats_info}
        return info, stats_info

@jax.jit
def _update_discriminator_jit(
    *,
    expert_batch: jnp.ndarray,
    learner_batch: jnp.ndarray,
    disc: GAILDiscriminator,
):
    new_disc, disc_info, disc_stats_info = disc.update(expert_batch=expert_batch, learner_batch=learner_batch)
    return new_disc, disc_info, disc_stats_info
