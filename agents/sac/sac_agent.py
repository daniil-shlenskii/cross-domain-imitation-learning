from copy import deepcopy
from typing import Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from agents.sac.losses import (actor_loss_fn, critic_loss_fn,
                               temperature_loss_fn)
from nn.train_state import TrainState
from utils.types import DataType, Params
from utils.utils import instantiate_optimizer


class SACAgent(Agent): 
    actor: TrainState
    critic1: TrainState
    critic2: TrainState
    target_critic1_params: Params
    target_critic2_params: Params
    temperature: TrainState
    target_entropy: float = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)

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
        actor_module_config: DictConfig,
        critic_module_config: DictConfig,
        temperature_module_config: DictConfig,
        #
        actor_optimizer_config: DictConfig,
        critic_optimizer_config: DictConfig,
        temperature_optimizer_config: DictConfig,
        #
        target_entropy: float = None,
        backup_entropy: bool = True,
        #
        discount: float = 0.99,
        tau: float = 0.005,
    ):
        # reproducibility keys
        rng = jax.random.key(seed)
        rng, actor_key, critic1_key, critic2_key, temperature_key = jax.random.split(rng, 5)

        # actor, critic and temperature initialization
        observation = np.ones(observation_dim)
        action = np.ones(action_dim)

        actor_module = instantiate(actor_module_config, action_dim=action_dim, low=low, high=high)
        critic1_module = instantiate(critic_module_config)
        critic2_module = instantiate(critic_module_config)
        temperature_module = instantiate(temperature_module_config)

        actor_params = actor_module.init(actor_key, observation)["params"]
        critic1_params = critic1_module.init(critic1_key, observation, action)["params"]
        critic2_params = critic2_module.init(critic2_key, observation, action)["params"]
        temperature_params = temperature_module.init(temperature_key)["params"]

        target_critic1_params = deepcopy(critic1_params)
        target_critic2_params = deepcopy(critic2_params)

        actor = TrainState.create(
            loss_fn=actor_loss_fn,
            apply_fn=actor_module.apply,
            params=actor_params,
            tx=instantiate_optimizer(actor_optimizer_config),
            info_key="actor",
        )
        critic1 = TrainState.create(
            loss_fn=critic_loss_fn,
            apply_fn=critic1_module.apply,
            params=critic1_params,
            tx=instantiate_optimizer(critic_optimizer_config),
            info_key="critic1",
        )
        critic2 = TrainState.create(
            loss_fn=critic_loss_fn,
            apply_fn=critic2_module.apply,
            params=critic2_params,
            tx=instantiate_optimizer(critic_optimizer_config),
            info_key="critic2",
        )
        temperature = TrainState.create(
            loss_fn=temperature_loss_fn,
            apply_fn=temperature_module.apply,
            params=temperature_params,
            tx=instantiate_optimizer(temperature_optimizer_config),
            info_key="temperature",
        )

        # target entropy init
        action_dim = action.shape[-1]
        if target_entropy is None:
            target_entropy = -action_dim

        return cls(
            rng=rng,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            temperature=temperature,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            discount=discount,
            tau=tau,
            _save_attrs = (
                "actor",
                "critic1",
                "critic2",
                "temperature",
                "target_critic1_params",
                "target_critic2_params",
            ),
        )

    @property
    def critic(self):
        return self.critic1

    def update(self, batch: DataType):
        new_agent, info, stats_info = _update_jit(batch=batch, agent=self)
        return new_agent, info, stats_info

@jax.jit
def _update_jit(
    *,
    batch: DataType,
    agent: Agent,
):
    # temperature
    temp = agent.temperature()

    # critic target
    new_rng, key = jax.random.split(agent.rng)
    dist = agent.actor(batch["observations_next"])
    actions_next, log_prob_next = dist.sample_and_log_prob(seed=key)

    state_action_value_next = jnp.minimum(
        agent.critic1.apply_fn({"params": agent.target_critic1_params}, batch["observations_next"], actions_next),
        agent.critic2.apply_fn({"params": agent.target_critic2_params}, batch["observations_next"], actions_next)
    )

    critic_target = batch["rewards"] + (1 - batch["dones"]) * agent.discount * state_action_value_next
    if agent.backup_entropy:
        critic_target -= (1 - batch["dones"]) * agent.discount * temp * log_prob_next

    # critic update
    new_critic1, critic1_info, critic1_stats_info = agent.critic1.update(batch=batch, target=critic_target)
    new_critic2, critic2_info, critic2_stats_info = agent.critic2.update(batch=batch, target=critic_target)

    # actor update
    new_rng, key = jax.random.split(new_rng)
    new_actor, actor_info, actor_stats_info = agent.actor.update(batch=batch, critic1=new_critic1, critic2=new_critic2, temp=temp, key=key)

    # temperature update
    new_temperature, temperature_info, temperature_stats_info = agent.temperature.update(entropy=actor_info["entropy"], target_entropy=agent.target_entropy)

    # target_critic update
    new_target_critic1_params = _update_target_net(new_critic1.params, agent.target_critic1_params, agent.tau)
    new_target_critic2_params = _update_target_net(new_critic2.params, agent.target_critic2_params, agent.tau)

    new_agent = agent.replace(
        rng=new_rng,
        actor=new_actor,
        critic1=new_critic1,
        critic2=new_critic2,
        target_critic1_params=new_target_critic1_params,
        target_critic2_params=new_target_critic2_params,
        temperature=new_temperature,
    )
    info = {**critic1_info, **critic2_info, **actor_info, **temperature_info}
    stats_info = {**critic1_stats_info, **critic2_stats_info, **actor_stats_info, **temperature_stats_info}
    return new_agent, info, stats_info

def _update_target_net(
    online_params: Params, target_params: Params, tau: float
) -> Tuple[TrainState, Dict[str, float]]:
    new_target_params = jax.tree.map(
        lambda t1, t2: t1 * tau + t2 * (1 - tau),
        online_params, target_params,
    )
    return new_target_params
