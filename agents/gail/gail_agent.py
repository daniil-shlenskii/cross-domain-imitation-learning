import functools

import gymnasium as gym
import jax
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from agents.gail.gail_discriminator import GAILDiscriminator
from utils import (get_buffer_state_size, instantiate_jitted_fbx_buffer,
                   load_pickle)
from utils.types import Buffer, BufferState, DataType


class GAILAgent(Agent):
    agent: Agent
    policy_discriminator: GAILDiscriminator
    expert_buffer: Buffer = struct.field(pytree_node=False)
    expert_buffer_state: BufferState = struct.field(pytree_node=False)
    n_policy_discriminator_updates: int = struct.field(pytree_node=False)

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
        policy_discriminator_config: DictConfig,
        #
        n_policy_discriminator_updates: int = 1,
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
        expert_buffer_state = load_pickle(expert_buffer_state_path)

        buffer_state_size = get_buffer_state_size(expert_buffer_state)
        expert_buffer_state_exp = expert_buffer_state.experience
        new_expert_buffer_state_exp = {}
        for k, v in expert_buffer_state_exp.items():
            new_expert_buffer_state_exp[k] = v[0, :buffer_state_size]
        expert_buffer_state.replace(
            experience=new_expert_buffer_state_exp,
            current_index=0,
            is_full=True,
        )

        expert_buffer = instantiate_jitted_fbx_buffer({
            "_target_": "flashbax.make_item_buffer",
            "sample_batch_size": expert_batch_size,
            "min_length": buffer_state_size,
            "max_length": buffer_state_size,
            "add_batches": False,
        })

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
            n_policy_discriminator_updates=n_policy_discriminator_updates,
            _save_attrs = _save_attrs,
            **kwargs,
        )

    @property
    def actor(self):
        return self.agent.actor

    def update(self, batch: DataType):
        update_agent = bool(
                (self.policy_discriminator.state.step + 1) % self.n_policy_discriminator_updates == 0
        )
        new_gail_agent, info, stats_info = _update_jit(
            batch, gail_agent=self, update_agent=update_agent
        )
        return new_gail_agent, info, stats_info

    @functools.partial(jax.jit, static_argnames="update_agent")
    def update_gail(
        self,
        batch: DataType,
        expert_batch: DataType,
        policy_discriminator_learner_batch: DataType,
        update_agent: bool,
    ):
        new_params = {}

        # update policy_discriminator
        new_disc, info, stats_info = self.policy_discriminator.update(
            learner_batch=policy_discriminator_learner_batch,
            expert_batch=expert_batch,
        )
        new_params["policy_discriminator"] = new_disc

        # update agent
        if update_agent:
            batch["rewards"] = new_disc.get_rewards(batch)
            new_agent, agent_info, agent_stats_info = self.agent.update(batch)
            new_params["agent"] = new_agent
            info.update(agent_info)
            stats_info.update(agent_stats_info)

        new_gail_agent = self.replace(**new_params)
        return new_gail_agent, info, stats_info

@functools.partial(jax.jit, static_argnames="update_agent")
def _update_jit(
    batch: DataType,
    gail_agent: GAILAgent,
    update_agent: bool,
):
    # sample expert batch
    new_rng, key = jax.random.split(gail_agent.rng)
    expert_batch = gail_agent.expert_buffer.sample(gail_agent.expert_buffer_state, key).experience
    new_gail_agent = gail_agent.replace(rng=new_rng)

    # update agent and policy policy_discriminator
    new_gail_agent, info, stats_info = new_gail_agent.update_gail(
        batch=batch,
        expert_batch=expert_batch,
        policy_discriminator_learner_batch=batch,
        update_agent=update_agent,
    )
    return new_gail_agent, info, stats_info
