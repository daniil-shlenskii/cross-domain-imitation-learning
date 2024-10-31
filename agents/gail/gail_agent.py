import jax

import gymnasium as gym

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from nn.train_state import TrainState
from utils.types import DataType, PRNGKey, Buffer


class GAILAgent(Agent):
    def __init__(
        self,
        *
        seed: int,
        observations_space: gym.Space,
        action_space: gym.Space,
        #
        expert_buffer: Buffer,
        #
        agent_config: DictConfig,
        discriminator_config: DictConfig
    ):
        # reproducibility keys
        rng = jax.random.key(seed)
        self._rng = rng

        # agent and discriminator init
        self.agent = instantiate(agent_config, observations_space=observations_space, action_space=action_space)
        self.discriminator = instantiate(discriminator_config)

        # 
        self.seed = seed
        self.expert_buffer = expert_buffer

    @property
    def actor(self):
        return self.agent.actor

    def update(self, batch: DataType):
        ...

@jax.jit
def _update_jit(
    rng: PRNGKey,
    batch: DataType,
    expert_batch: DataType,
    #
    agent: Agent,
    disc: GAILDiscriminator,
):
    batch["rewards"] = disc.get_rewards(batch["observations"], batch["observations_next"])
    new_agent, agent_info, agent_stats_info = agent.update(batch)
    new_disc, disc_info, disc_stats_info = disc.update(expert_batch)

    info = {**agent_info, **disc_info}
    stats_info = {**agent_stats_info, **disc_stats_info}
    return (
        rng,
        new_agent,
        new_disc,
        info,
        stats_info,
    )
