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
from agents.dida.utils import process_policy_discriminator_input
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
        sar_p: float = 0.66,
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
            learner_encoder=learner_encoder,
            expert_encoder=expert_encoder,
            domain_discriminator=domain_discriminator,
            sar_p=sar_p,
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

    def update(self, batch: DataType):
        self.rng, expert_key, anchor_key = jax.random.split(self.rng, 3)
        expert_batch = self.expert_buffer.sample(self.expert_buffer_state, expert_key).experience
        anchor_batch = self.expert_buffer.sample(self.anchor_buffer_state, anchor_key).experience

        info, stats_info = {}, {}
        # update encoders
        (
            self.learner_encoder,
            self.expert_encoder,
            self.domain_discriminator,
            encoded_learner_state,
            encoded_expert_state,
            info,
            stats_info,
        ) = _update_encoders_part_jit(
            learner_batch=batch,
            expert_batch=expert_batch,
            learner_encoder=self.learner_encoder,
            expert_encoder=self.expert_encoder,
            policy_discriminator=self.discriminator,
            domain_discriminator=self.domain_discriminator,
        )

        # prepare gail input
        # embed observation
        encoders_dim = encoded_learner_state.shape[-1] // 2
        batch["observations"] = encoded_learner_state[:, :encoders_dim]
        batch["observations_next"] = encoded_learner_state[:, encoders_dim:]
        expert_batch["observations"] = encoded_expert_state[:, :encoders_dim]
        expert_batch["observations_next"] = encoded_expert_state[:, encoders_dim:]
        anchor_batch["observations"] = self.expert_encoder(self.expert_encoder, anchor_batch["observations"])
        anchor_batch["observations_next"] = self.expert_encoder(self.expert_encoder, anchor_batch["observations_next"])

        # get das alph param with sar
        alpha = self_adaptive_rate(
            domain_discriminator=self.domain_discriminator,
            learner_batch=batch,
            expert_batch=expert_batch,
            p=self.sar_p,
        )

        # get gail learner batch via das
        self.rng, batch = domain_adversarial_sampling(
            rng=self.rng,
            embedded_learner_batch=batch,
            embedded_anchor_batch=anchor_batch,
            domain_discriminator=self.domain_discriminator,
            alpha=alpha,
        )

        # apply gail
        gail_info, gail_stats_info = super().update(batch=batch, expert_batch=expert_batch)

        info.update({**gail_info})
        stats_info.update({**gail_stats_info})
        return info, stats_info

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.learner_encoder(observations)

@jax.jit
def _update_encoders_part_jit(
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

    # store batch encoding for policy discriminator update
    encoded_learner_policy_batch = learner_encoder_info["generations"]
    encoded_expert_policy_batch = expert_encoder_info["generations"]

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

    # update domain discriminator
    new_domain_disc, domain_disc_info, domain_disc_stats_info = domain_discriminator.update(
        real_batch=learner_encoder_info.pop("generations"),
        fake_batch=expert_encoder_info.pop("generations")
    )

    info["upd_with_policy_disc"] = {**learner_encoder_info, **expert_encoder_info, **domain_disc_info}
    stats_info["upd_with_policy_disc"] = {**learner_encoder_stats_info, **expert_encoder_stats_info, **domain_disc_stats_info}
    
    info.update({**learner_encoder_info, **expert_encoder_info, **domain_disc_info})
    stats_info.update({**learner_encoder_stats_info, **expert_encoder_stats_info, **domain_disc_stats_info})

    return (
        new_learner_encoder,
        new_expert_encoder,
        new_domain_disc,
        encoded_learner_policy_batch,
        encoded_expert_policy_batch,
        info,
        stats_info,
    )
