import jax

import gymnasium as gym

from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from nn.train_state import TrainState
from utils.types import DataType, PRNGKey


class GAILAgent(Agent):
    def __init__(
        self,
        *
        seed: int,
        observations_space: gym.Space,
        actions_space: gym.Space,
        #
        expert_buffer: Buffer,
        #
        agent_config: DictConfig,
        discriminator_config: DictConfig
    ):
        # reproducibility keys
        rng = jax.random.key(seed)
        self._rng = rng

        # agent initialization
        
  
    def update(self, batch: DataType):
        ...


@jax.jit
def _update_jit(
    batch: DataType,
    rng: PRNGKey,
    #
    agent: Agent,
    discriminator: TrainState,
):
