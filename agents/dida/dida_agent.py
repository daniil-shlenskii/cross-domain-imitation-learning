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
from agents.gail.gail_discriminator import GAILDiscriminator
from gan.discriminator import Discriminator
from gan.generator import Generator
from nn.train_state import TrainState
from utils.types import Buffer, BufferState, DataType
from utils.utils import get_buffer_state_size, load_pickle


class DIDA(GAILAgent):
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
        learner_encoder_config: DictConfig,
        expert_encoder_config: DictConfig,
        domain_discriminator_config: DictConfig,
    ):  
        discriminator_config["info_key"] = "policy_discriminator"

        learner_encoder_input_sample = jnp.ones(observation_dim)
        learner_encoder = instantiate(
            learner_encoder_config,
            seed=seed,
            input_sample=learner_encoder_input_sample,
            info_key="learner_encoder",
            _recursive_=False,
        )

        expert_buffer_state = load_pickle(expert_buffer_state_path)
        expert_observation_dim = expert_buffer_state.experience["observation"].shape[-1]
        expert_encoder_input_sample = jnp.ones(expert_observation_dim)
        expert_encoder = instantiate(
            expert_encoder_config,
            seed=seed,
            input_sample=expert_encoder_input_sample,
            info_key="expert_encoder",
            _recursive_=False,
        )

        new_observation_dim = learner_encoder(learner_encoder_input_sample).shape[-1]
        obs = np.ones(new_observation_dim)
        domain_discriminator_input_sample = jnp.concatenate([obs, obs], axis=-1)
        domain_discriminator = instantiate(
            domain_discriminator_config,
            seed=seed,
            input_sample=domain_discriminator_input_sample,
            info_key="domain_discriminator",
            _recursive_=False,
        )

        return super().create(
            seed=seed,
            observation_dim=new_observation_dim,
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
        anchor_buffer_state["observations_next"].at[0].set(
            anchor_buffer_state["observations_next"][0:, perm_idcs]
        )
        self.anchor_buffer_state = anchor_buffer_state

    def update(self, batch: DataType):
        self.rng, expert_key, anchor_key = jax.random.split(self.rng, 3)
        expert_batch = self.expert_buffer.sample(self.expert_buffer_state, expert_key).experience
        anchor_batch = self.expert_buffer.sample(self.anchor_buffer_state, anchor_key).experience

        # update encoders
        (
            self.learner_encoder,
            self.expert_encoder,
            self.domain_discriminator,
            info,
            stats_info,
        ) = _update_encoders_part_jit(
            learner_batch=batch,
            expert_batch=expert_batch,
            anchor_batch=anchor_batch,
            learner_encoder=self.learner_encoder,
            expert_encoder=self.expert_encoder,
            policy_discriminator=self.discriminator,
            domain_discriminator=self.domain_discriminator,
        )

        # prepare gail input
        gail_input_kwargs = {
            "batch": deepcopy(batch),
            "expert_batch": deepcopy(expert_batch)
        }
        for key_to_change in ["observations", "observations_next"]:
            gail_input_kwargs["batch"][key_to_change] = self.learner_encoder(gail_input_kwargs["batch"][key_to_change])
            gail_input_kwargs["expert_batch"][key_to_change] = self.expert_encoder(gail_input_kwargs["expert_batch"][key_to_change])(batch)

        # apply gail
        gail_info, gail_stats_info = super().update(**gail_input_kwargs)

        info["gail_update"] = {**gail_info}
        stats_info["gail_update"] = {**gail_stats_info}
        return info, stats_info

@jax.jit
def _update_encoders_part_jit(
    learner_batch: DataType,
    expert_batch: DataType,
    learner_encoder: Generator,
    expert_encoder: Generator,
    policy_discriminator: Discriminator,
    domain_discriminator: Discriminator,
):
    # update encoders via policy discriminator
    learner_policy_batch = jnp.stack([learner_batch["observations"], learner_batch["observations_next"]])
    expert_policy_batch = jnp.stack([expert_batch["observations"], expert_batch["observations_next"]])
    
    new_learner_encoder, learner_encoder_info, learner_encoder_stats_info = learner_encoder.update(
        batch=learner_policy_batch, discriminator=policy_discriminator, process_discriminator_input=_process_policy_discriminator_input 
    )
    new_expert_encoder, expert_encoder_info, expert_encoder_stats_info = expert_encoder.update(
        batch=expert_policy_batch, discriminator=policy_discriminator, process_discriminator_input=_process_policy_discriminator_input 
    )

    del learner_encoder_info["generations"], expert_encoder_info["generations"]

    info = {"policy_update": {**learner_encoder_info, **expert_encoder_info}}
    stats_info = {"policy_update": {**learner_encoder_stats_info, **expert_encoder_stats_info}}

    # update encoders via domain discriminator
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
        real_batch=learner_encoder_info["generation"],
        fake_batch=expert_encoder_info["generation"]
    )

    info["domain_update"] = {**learner_encoder_info, **expert_encoder_info, **domain_disc_info}
    stats_info["domain_update"] = {**learner_encoder_stats_info, **expert_encoder_stats_info, **domain_disc_stats_info}

    return (
        new_learner_encoder,
        new_expert_encoder,
        new_domain_disc,
        info,
        stats_info,
    )

def _process_policy_discriminator_input(x):
    doubled_b_size, dim = x.shape
    return lambda x: x.reshape(2, doubled_b_size // 2, dim).transpose(1, 2, 0)