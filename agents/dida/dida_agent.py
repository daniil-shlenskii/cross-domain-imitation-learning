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
from utils.utils import load_pickle


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
        learner_encoder_input_sample = jnp.ones(observation_dim)
        learner_encoder = instantiate(
            learner_encoder_config,
            seed=seed,
            input_sample=learner_encoder_input_sample,
            _recursive_=False,
        )

        expert_buffer_state = load_pickle(expert_buffer_state_path)
        expert_observation_dim = expert_buffer_state.experience["observation"].shape[-1]
        expert_encoder_input_sample = jnp.ones(expert_observation_dim)
        expert_encoder = instantiate(
            expert_encoder_config,
            seed=seed,
            input_sample=expert_encoder_input_sample,
            _recursive_=False,
        )

        new_observation_dim = learner_encoder(learner_encoder_input_sample).shape[-1]
        obs = np.ones(new_observation_dim)
        domain_discriminator_input_sample = jnp.concatenate([obs, obs], axis=-1)
        domain_discriminator = instantiate(
            domain_discriminator_config,
            seed=seed,
            input_sample=domain_discriminator_input_sample,
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
        learner_enocder: Generator,
        expert_enocder: Generator,
        domain_discriminator: Discriminator,
    ):
        ...