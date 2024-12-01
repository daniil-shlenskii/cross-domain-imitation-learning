import functools
from typing import Dict

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
from utils.types import DataType
from utils.utils import instantiate_jitted_fbx_buffer, load_pickle


class GAILAgent(Agent):
    agent: Agent
    discriminator: GAILDiscriminator
    expert_buffer: Buffer = struct.field(pytree_node=False)
    expert_buffer_state: BufferState = struct.field(pytree_node=False)
    n_discriminator_updates: int = struct.field(pytree_node=False)

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
        n_discriminator_updates: int = 1,
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

        discriminator = instantiate(
            discriminator_config,
            seed=seed,
            input_dim=observation_dim * 2,
            _recursive_=False,
        )

        # expert buffer init
        expert_buffer_state = load_pickle(expert_buffer_state_path)
        expert_buffer = instantiate_jitted_fbx_buffer(
            fbx_buffer_config=dict(
                _target_="flashbax.make_item_buffer",
                sample_batch_size=expert_batch_size,
                min_length=expert_buffer_state.current_index,
                max_length=expert_buffer_state.current_index,
                add_batches=False,
            )
        )

        _save_attrs = kwargs.pop(
            "_save_attrs",
            ("agent", "discriminator")
        )

        return cls(
            rng=jax.random.key(seed),
            expert_buffer=expert_buffer,
            expert_buffer_state=expert_buffer_state,
            agent=agent,
            discriminator=discriminator,
            n_discriminator_updates=n_discriminator_updates,
            _save_attrs = _save_attrs,
            **kwargs,
        )

    @property
    def actor(self):
        return self.agent.actor

    def update(self, batch: DataType):
        train_agent = bool(
                self.discriminator.state.step % self.n_discriminator_updates == 0
        )
        new_agent, info, stats_info = _update_jit(
            batch, gail_agent=self, train_agent=train_agent
        )
        return new_agent, info, stats_info

@functools.partial(jax.jit, static_argnames="train_agent")
def _update_jit(
    batch: DataType,
    gail_agent: GAILAgent,
    train_agent: bool,
):
    new_params = {}
    
    # sample expert batch
    new_rng, key = jax.random.split(gail_agent.rng)
    expert_batch = gail_agent.expert_buffer.sample(gail_agent.expert_buffer_state, key).experience
    new_params["rng"] = new_rng
   
    # update discriminator
    new_disc, info, stats_info = gail_agent.discriminator.update(
        learner_batch=batch,
        expert_batch=expert_batch,
    )
    new_params["discriminator"] = new_disc

    # update agent
    if train_agent:
        batch["rewards"] = new_disc.get_rewards(batch)
        new_agent, agent_info, agent_stats_info = gail_agent.agent.update(batch)

        new_params["agent"] = new_agent
        info.update(agent_info)
        stats_info.update(agent_stats_info)

    new_agent = gail_agent.replace(**new_params)
    return new_agent, info, stats_info

