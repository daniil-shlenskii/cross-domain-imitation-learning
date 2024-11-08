from copy import deepcopy
from typing import Tuple

import flashbax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents import GAILAgent
from agents.base_agent import Agent
from agents.dida.das import domain_adversarial_sampling
from agents.dida.sar import self_adaptive_rate
from agents.dida.utils import (encode_observation,
                               process_policy_discriminator_input)
from agents.gail.gail_discriminator import GAILDiscriminator
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils.types import Buffer, BufferState, DataType, PRNGKey
from utils.utils import get_buffer_state_size, load_pickle


class DIDAAgent(GAILAgent):
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
        encoders_dim: int,
        #
        expert_batch_size: int,
        expert_buffer_state_path: str,
        #
        agent_config: DictConfig,
        discriminator_config: DictConfig,
        #
        learner_encoder_config: DictConfig,
        expert_encoder_config: DictConfig,
        domain_discriminator_config: DictConfig,
        #
        sar_p: float = 0.5,
        #
        n_domain_discriminator_updates: int = 1,
    ):  
        discriminator_config["info_key"] = "policy_discriminator"

        learner_encoder_input_sample = jnp.ones(observation_dim)
        learner_encoder = instantiate(
            learner_encoder_config,
            seed=seed,
            input_sample=learner_encoder_input_sample,
            output_dim=encoders_dim,
            info_key="learner_encoder",
            _recursive_=False,
        )

        expert_buffer_state = load_pickle(expert_buffer_state_path)
        expert_observation_dim = expert_buffer_state.experience["observations"].shape[-1]
        expert_encoder_input_sample = jnp.ones(expert_observation_dim)
        expert_encoder = instantiate(
            expert_encoder_config,
            seed=seed,
            input_sample=expert_encoder_input_sample,
            output_dim=encoders_dim,
            info_key="expert_encoder",
            _recursive_=False,
        )

        domain_discriminator_input_sample = np.ones(encoders_dim)
        domain_discriminator = instantiate(
            domain_discriminator_config,
            seed=seed,
            input_sample=domain_discriminator_input_sample,
            info_key="domain_discriminator",
            _recursive_=False,
        )

        return super().create(
            seed=seed,
            observation_dim=encoders_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            expert_batch_size=expert_batch_size,
            expert_buffer_state_path=expert_buffer_state_path,
            agent_config=agent_config,
            discriminator_config=discriminator_config,
            #
            learner_encoder=learner_encoder,
            expert_encoder=expert_encoder,
            domain_discriminator=domain_discriminator,
            sar_p=sar_p,
            n_domain_discriminator_updates=n_domain_discriminator_updates
        )
    
    def __init__(
        self,
        *,
        seed: int,
        expert_buffer: Buffer,
        expert_buffer_state: BufferState,
        agent: Agent,
        discriminator: Discriminator,
        learner_encoder: Generator,
        expert_encoder: Generator,
        domain_discriminator: Discriminator,
        sar_p: float,
        n_domain_discriminator_updates: int,
    ):
        self.rng = jax.random.key(seed=seed)
        self.expert_buffer = expert_buffer
        self.expert_buffer_state = expert_buffer_state
        self.agent = agent
        self.discriminator = discriminator
        self.learner_encoder = learner_encoder
        self.expert_encoder = expert_encoder
        self.domain_discriminator = domain_discriminator

        anchor_buffer_state = deepcopy(expert_buffer_state)
        buffer_state_size = get_buffer_state_size(anchor_buffer_state)
        perm_idcs = np.random.permutation(buffer_state_size)
        anchor_buffer_state.experience["observations_next"].at[0, :buffer_state_size].set(
            anchor_buffer_state.experience["observations_next"][0, :buffer_state_size][perm_idcs]
        )
        self.anchor_buffer_state = anchor_buffer_state

        self.sar_p = sar_p

        self.n_domain_discriminator_updates = n_domain_discriminator_updates

    def update(self, batch: DataType):
        (
            self.rng,
            self.learner_encoder,
            self.expert_encoder,
            self.policy_disc,
            self.domain_disc,
            self.agent,
            info,
            stats_info
        ) = _update_jit(
            rng=self.rng,
            batch=batch,
            expert_buffer=self.expert_buffer,
            expert_buffer_state=self.expert_buffer_state,
            anchor_buffer_state=self.anchor_buffer_state,
            learner_encoder=self.learner_encoder,
            expert_encoder=self.expert_encoder,
            policy_discriminator=self.discriminator,
            domain_discriminator=self.domain_discriminator,
            agent=self.agent,
            sar_p=self.sar_p,
        )
        return info, stats_info

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return encode_observation(self.learner_encoder, observations)
    
@jax.jit
def _update_jit(
    *,
    rng: PRNGKey,
    batch: DataType,
    expert_buffer: Buffer,
    expert_buffer_state: BufferState,
    anchor_buffer_state: BufferState,
    #
    learner_encoder: Generator,
    expert_encoder: Generator,
    policy_discriminator: Discriminator,
    domain_discriminator: Discriminator,
    agent: Agent,
    #
    sar_p: float
):
    new_rng, expert_key, anchor_key = jax.random.split(rng, 3)
    expert_batch = expert_buffer.sample(expert_buffer_state, expert_key).experience
    anchor_batch = expert_buffer.sample(anchor_buffer_state, anchor_key).experience

    # UPDATE encoders
    (
        new_learner_encoder,
        new_expert_encoder,
        encoded_learner_domain_batch,
        encoded_expert_domain_batch,
        encoded_learner_policy_batch,
        encoded_expert_policy_batch,
        encoders_info,
        encoders_stats_info,
    ) = _update_encoders(
        learner_batch=batch,
        expert_batch=expert_batch,
        learner_encoder=learner_encoder,
        expert_encoder=expert_encoder,
        policy_discriminator=policy_discriminator,
        domain_discriminator=domain_discriminator,
    )

    # UPDATE domain discriminator
    new_domain_disc, domain_disc_info, domain_disc_stats_info = domain_discriminator.update(
        real_batch=encoded_learner_domain_batch,
        fake_batch=encoded_expert_domain_batch
    )

    # UPDATE policy discriminator and agent with gail
    # encode observation
    encoders_dim = encoded_learner_domain_batch.shape[-1]
    batch["observations"] = encoded_learner_policy_batch[:, :encoders_dim]
    batch["observations_next"] = encoded_learner_policy_batch[:, encoders_dim:]
    expert_batch["observations"] = encoded_expert_policy_batch[:, :encoders_dim]
    expert_batch["observations_next"] = encoded_expert_policy_batch[:, encoders_dim:]
    anchor_batch["observations"] = expert_encoder(anchor_batch["observations"])
    anchor_batch["observations_next"] = expert_encoder(anchor_batch["observations_next"])

    # get das alpha param with sar
    alpha = self_adaptive_rate(
        domain_discriminator=domain_discriminator,
        learner_batch=batch,
        expert_batch=expert_batch,
        p=sar_p,
    )

    # get gail learner batch via das
    rng, mixed_batch = domain_adversarial_sampling(
        rng=rng,
        embedded_learner_batch=batch,
        embedded_anchor_batch=anchor_batch,
        domain_discriminator=domain_discriminator,
        alpha=alpha,
    )

    # apply gail
    agent, new_policy_disc, gail_info, gail_stats_info = _update_gail(
        policy_batch=mixed_batch,
        expert_policy_batch=expert_batch,
        agent=agent,
        discriminator=policy_discriminator,
    )


    info = {**encoders_info, **domain_disc_info, **gail_info}
    stats_info = {**encoders_stats_info, **domain_disc_stats_info, **gail_stats_info}
    return (
        new_rng,
        new_learner_encoder,
        new_expert_encoder,
        new_policy_disc,
        new_domain_disc,
        agent,
        info,
        stats_info
    )


def _update_encoders(
    learner_batch: DataType,
    expert_batch: DataType,
    learner_encoder: Generator,
    expert_encoder: Generator,
    policy_discriminator: Discriminator,
    domain_discriminator: Discriminator,
):
    # update encoders with policy discriminator
    learner_policy_batch = jnp.concatenate([learner_batch["observations"], learner_batch["observations_next"]], axis=0)
    expert_policy_batch = jnp.concatenate([expert_batch["observations"], expert_batch["observations_next"]], axis=0)
    
    new_learner_encoder, learner_encoder_info, learner_encoder_stats_info = learner_encoder.update(
        hold_grad=True,
        batch=learner_policy_batch,
        discriminator=policy_discriminator,
        process_discriminator_input=process_policy_discriminator_input 
    )
    new_expert_encoder, expert_encoder_info, expert_encoder_stats_info = expert_encoder.update(
        hold_grad=True,
        batch=expert_policy_batch,
        discriminator=policy_discriminator,
        process_discriminator_input=process_policy_discriminator_input 
    )

    # store encoded batch for policy discriminator update
    encoded_learner_policy_batch = learner_encoder_info.pop("generations")
    encoded_expert_policy_batch = expert_encoder_info.pop("generations")

    info = {"upd_with_policy_disc": {**learner_encoder_info, **expert_encoder_info}}
    stats_info = {"upd_with_policy_disc": {**learner_encoder_stats_info, **expert_encoder_stats_info}}

    # update encoders with domain discriminator
    learner_domain_batch = learner_batch["observations"]
    expert_domain_batch = expert_batch["observations"]

    new_learner_encoder, learner_encoder_info, learner_encoder_stats_info = new_learner_encoder.update(
        batch=learner_domain_batch, discriminator=domain_discriminator
    )
    new_expert_encoder, expert_encoder_info, expert_encoder_stats_info = new_expert_encoder.update(
        batch=expert_domain_batch, discriminator=domain_discriminator
    )

    # store encoded batch for domain discriminator update
    encoded_learner_domain_batch = learner_encoder_info.pop("generations")
    encoded_expert_domain_batch = expert_encoder_info.pop("generations")

    info["upd_with_domain_disc"] = {**learner_encoder_info, **expert_encoder_info}
    stats_info["upd_with_domain_disc"] = {**learner_encoder_stats_info, **expert_encoder_stats_info}

    return (
        new_learner_encoder,
        new_expert_encoder,
        encoded_learner_domain_batch,
        encoded_expert_domain_batch,
        encoded_learner_policy_batch,
        encoded_expert_policy_batch,
        info,
        stats_info,
    )

def _update_gail(
    *,
    policy_batch: DataType,
    expert_policy_batch: DataType,
    #
    agent: Agent,
    discriminator: GAILDiscriminator,
):
    # update agent
    policy_batch["reward"] = discriminator.get_rewards(policy_batch)
    agent_info, agent_stats_info = agent.update(policy_batch)

    # update discriminator
    new_disc, disc_info, disc_stats_info = discriminator.update(learner_batch=policy_batch, expert_batch=expert_policy_batch)

    info = {**agent_info, **disc_info}
    stats_info = {**agent_stats_info, **disc_stats_info}
    return agent, new_disc, info, stats_info