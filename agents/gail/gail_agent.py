import functools

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import wandb
from agents.base_agent import Agent
from agents.gail.gail_discriminator import GAILDiscriminator
from gan.discriminator import Discriminator
from utils import (convert_figure_to_array, get_buffer_state_size,
                   instantiate_jitted_fbx_buffer, load_pickle, sample_batch)
from utils.types import Buffer, BufferState, DataType

from .utils import get_sample_discriminator_hists


class GAILAgent(Agent):
    agent: Agent
    policy_discriminator: GAILDiscriminator
    sample_discriminator: Discriminator
    expert_batch_size: int = struct.field(pytree_node=False)
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
        sample_discriminator_config: DictConfig = None,
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

        if sample_discriminator_config is not None:
            sample_discriminator = instantiate(
                sample_discriminator_config,
                seed=seed,
                input_dim=observation_dim,
                _recursive_=False,
            )
        else:
            sample_discriminator = None

        # expert buffer init
        _expert_buffer_state = load_pickle(expert_buffer_state_path)
        buffer_state_size = get_buffer_state_size(_expert_buffer_state)

        expert_buffer = instantiate_jitted_fbx_buffer({
            "_target_": "flashbax.make_prioritised_flat_buffer",
            "sample_batch_size": expert_batch_size,
            "min_length": 1,
            "max_length": buffer_state_size,
        })

        expert_buffer_state_exp = _expert_buffer_state.experience
        expert_buffer_state = expert_buffer.init({
            k: v[0] for k, v in expert_buffer_state_exp.items()
        })

        new_expert_buffer_state_exp = {}
        for k, v in expert_buffer_state_exp.items():
            new_expert_buffer_state_exp[k] = jnp.asarray(v[0, :buffer_state_size][None])
        expert_buffer_state = expert_buffer_state.replace(
            experience=new_expert_buffer_state_exp,
            current_index=0,
            is_full=True,
        )

        _save_attrs = kwargs.pop(
            "_save_attrs",
            ("agent", "policy_discriminator")
        )

        return cls(
            rng=jax.random.key(seed),
            expert_batch_size=expert_batch_size,
            expert_buffer=expert_buffer,
            expert_buffer_state=expert_buffer_state,
            agent=agent,
            policy_discriminator=policy_discriminator,
            sample_discriminator=sample_discriminator,
            n_policy_discriminator_updates=n_policy_discriminator_updates,
            _save_attrs = _save_attrs,
            **kwargs,
        )

    @property
    def actor(self):
        return self.agent.actor

    @jax.jit
    def _sample_expert_batch(self):
        rng = self.rng

        # sample random batch
        rng, random_batch = sample_batch(self.rng, self.expert_buffer, self.expert_buffer_state)

        # sample batch using sample_discriminator
        if self.sample_discriminator is not None:
            expert_experience = self.expert_buffer_state.experience
            expert_states = expert_experience["observations"][0]
            expert_states_logits = self.sample_discriminator(expert_states)
            shifted_logits = expert_states_logits - expert_states_logits.min()
            normalized_logits = shifted_logits / shifted_logits.max()
            relevance_probs = 1 - normalized_logits
            relevance_sample_priorities = relevance_probs / relevance_probs.sum()
            relevant_state = self.expert_buffer.set_priorities(
                self.expert_buffer_state,
                jnp.arange(expert_states.shape[0]),
                relevance_sample_priorities
            )
            rng, relevant_batch = sample_batch(rng, self.expert_buffer, relevant_state)
            return rng, relevant_batch, random_batch

        return rng, random_batch, random_batch

    def update(self, batch: DataType):
        update_agent = bool(
                (self.policy_discriminator.state.step + 1) % self.n_policy_discriminator_updates == 0
        )
        new_gail_agent, info, stats_info = _update_jit(
            batch, gail_agent=self, update_agent=update_agent
        )
        return new_gail_agent, info, stats_info

    def evaluate(
        self,
        *,
        seed: int,
        env: gym.Env,
        num_episodes: int,
        #
        convert_to_wandb_type: bool = True,
        #
        return_trajs: bool = False,
    ):
        eval_info, trajs = super().evaluate(seed=seed, env=env, num_episodes=num_episodes, return_trajectories=True)

        # domain sample discriminator historgrams
        (
            state_learner_hist,
            state_expert_hist,
        ) = get_sample_discriminator_hists(
            dida_agent=self,
            learner_trajs=trajs,
        )
        if convert_to_wandb_type:
            state_learner_hist = wandb.Image(convert_figure_to_array(state_learner_hist), caption="Sample Discriminator Learner logits")
            state_expert_hist = wandb.Image(convert_figure_to_array(state_expert_hist), caption="Sample Discriminator Expert logits")
        eval_info["sample_learner_hist"] = state_learner_hist
        eval_info["sample_expert_hist"] = state_expert_hist

        if return_trajs:
            return eval_info, trajs
        return eval_info

    @functools.partial(jax.jit, static_argnames="update_agent")
    def update_gail(
        self,
        batch: DataType,
        expert_batch: DataType,
        policy_discriminator_learner_batch: DataType,
        sample_discriminator_batch: DataType,
        update_agent: bool,
    ):
        new_params = {}

        # update policy_discriminator
        new_domain_disc, info, stats_info = self.policy_discriminator.update(
            learner_batch=policy_discriminator_learner_batch,
            expert_batch=expert_batch,
        )
        new_params["policy_discriminator"] = new_domain_disc

        # update agent
        if update_agent:
            batch["rewards"] = new_domain_disc.get_rewards(batch)
            new_agent, agent_info, agent_stats_info = self.agent.update(batch)
            new_params["agent"] = new_agent
            info.update(agent_info)
            stats_info.update(agent_stats_info)

        # update sample discriminator if exists
        if self.sample_discriminator is not None:
            new_sample_disc, sample_disc_info, sample_disc_stats_info = self.sample_discriminator.update(
                real_batch=sample_discriminator_batch["observations"],
                fake_batch=batch["observations"],
            )
            new_params["sample_discriminator"] = new_sample_disc
            info.update(sample_disc_info)
            stats_info.update(sample_disc_stats_info)

        new_gail_agent = self.replace(**new_params)
        return new_gail_agent, info, stats_info

@functools.partial(jax.jit, static_argnames="update_agent")
def _update_jit(
    batch: DataType,
    gail_agent: GAILAgent,
    update_agent: bool,
):
    # sample expert batch
    new_rng, expert_batch, sample_discriminator_batch = gail_agent._sample_expert_batch()
    new_gail_agent = gail_agent.replace(rng=new_rng)

    # update agent and policy policy_discriminator
    new_gail_agent, info, stats_info = new_gail_agent.update_gail(
        batch=batch,
        expert_batch=expert_batch,
        policy_discriminator_learner_batch=batch,
        sample_discriminator_batch=sample_discriminator_batch,
        update_agent=update_agent,
    )
    return new_gail_agent, info, stats_info
