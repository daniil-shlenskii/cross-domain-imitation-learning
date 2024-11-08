import functools
from typing import Tuple

import flashbax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from agents.gail.gail_discriminator import GAILDiscriminator
from nn.train_state import TrainState
from utils.types import *
from utils.utils import load_pickle


class GAILAgent(Agent):
    agent: Agent
    discriminator: GAILDiscriminator
    expert_buffer: Buffer = struct.field(pytree_node=False)
    expert_buffer_state: BufferState = struct.field(pytree_node=False)

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
        **kwargs,
    ):
        rng = jax.random.key(seed)

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
            rng=rng,
            expert_buffer=expert_buffer,
            expert_buffer_state=expert_buffer_state,
            agent=agent,
            discriminator=discriminator,
            _save_attrs = (
                "agent",
                "discriminator"
            ),
            **kwargs,
        )

    @property
    def actor(self):
        return self.agent.actor

    def update(self, batch: DataType):
        (
            new_rng,
            new_agent,
            new_discriminator,
            info,
            stats_info,
        ) = _update_jit(
            rng=self.rng,
            batch=batch,
            expert_buffer=self.expert_buffer,
            expert_buffer_state=self.expert_buffer_state,
            agent=self.agent,
            discriminator=self.discriminator,
        )
        new_agent = self.replace(
            rng=new_rng,
            agent=new_agent,
            discriminator=new_discriminator
        )
        return new_agent, info, stats_info
    
@functools.partial(jax.jit, static_argnames="expert_buffer")
def _update_jit(
    *,
    rng: PRNGKey,
    batch: DataType,
    #
    expert_buffer: Buffer,
    expert_buffer_state: BufferState,
    # #
    agent: Agent,
    discriminator: GAILDiscriminator,
):
    new_rng, key = jax.random.split(rng)

    # process batch
    learner_batch = jnp.concatenate([batch["observations"], batch["observations_next"]], axis=-1)

    expert_batch = expert_buffer.sample(expert_buffer_state, key).experience
    expert_batch = jnp.concatenate([expert_batch["observations"], expert_batch["observations_next"]], axis=-1)

    # update agent
    batch["reward"] = discriminator.get_rewards(learner_batch)
    new_agent, agent_info, agent_stats_info = agent.update(batch)

    # update discriminator
    new_disc, disc_info, disc_stats_info = discriminator.update(learner_batch=learner_batch, expert_batch=expert_batch)

    info = {**agent_info, **disc_info}
    stats_info = {**agent_stats_info, **disc_stats_info}
    return new_rng, new_agent, new_disc, info, stats_info
