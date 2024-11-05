import functools
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.base_agent import Agent
from nn.train_state import TrainState
from utils.types import DataType, Params, PRNGKey
from utils.utils import instantiate_optimizer


class SACAgent(Agent):
    _non_train_state_attrs_to_save = [
        "target_critic1_params",
        "target_critic2_params",
    ]

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
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
        tau = 0.005,
    ):
        # reproducibility keys
        rng = jax.random.key(seed)
        actor_key, critic1_key, critic2_key, temperature_key = jax.random.split(rng, 4)

        # actor, critic and temperature initialization
        observation = observation_space.sample()
        action = action_space.sample()

        actor_module = cls.instantiate_actor_module(actor_module_config, action_space=action_space)
        critic1_module = instantiate(critic_module_config)
        critic2_module = instantiate(critic_module_config)
        temperature_module = instantiate(temperature_module_config)

        actor_params = actor_module.init(actor_key, observation)["params"]
        critic1_params = critic1_module.init(critic1_key, observation, action)["params"]
        critic2_params = critic2_module.init(critic2_key, observation, action)["params"]
        temperature_params = temperature_module.init(temperature_key)["params"]

        actor = TrainState.create(
            loss_fn=_actor_loss_fn,
            apply_fn=actor_module.apply,
            params=actor_params,
            tx=instantiate_optimizer(actor_optimizer_config),
            info_key="actor",
        )
        critic1 = TrainState.create(
            loss_fn=_critic_loss_fn,
            apply_fn=critic1_module.apply,
            params=critic1_params,
            tx=instantiate_optimizer(critic_optimizer_config),
            info_key="critic1",
        )
        critic2 = TrainState.create(
            loss_fn=_critic_loss_fn,
            apply_fn=critic2_module.apply,
            params=critic2_params,
            tx=instantiate_optimizer(critic_optimizer_config),
            info_key="critic2",
        )
        temperature = TrainState.create(
            loss_fn=_temperature_loss_fn,
            apply_fn=temperature_module.apply,
            params=temperature_params,
            tx=instantiate_optimizer(temperature_optimizer_config),
            info_key="temperature",
        )

        # target entropy init
        action_dim = action.shape[-1]
        if target_entropy is None:
            target_entropy = -action_dim
        else:
            target_entropy = target_entropy

        return cls(
            seed=seed,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            temperature=temperature,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            discount=discount,
            tau=tau,
        )

    def __init__(
        self,
        *,
        seed: int,
        #
        actor: TrainState,
        critic1: TrainState,
        critic2: TrainState,
        temperature: TrainState,
        #
        target_entropy: float,
        backup_entropy: bool,
        discount: float,
        tau: float,
    ):
        self.rng = jax.random.key(seed)

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.temperature = temperature

        self.target_critic1_params = deepcopy(critic1.params)
        self.target_critic2_params = deepcopy(critic2.params)

        self.target_entropy = target_entropy
        self.backup_entropy = backup_entropy
        self.discount = discount
        self.tau = tau

    @property
    def critic(self):
        return self.critic1

    def update(self, batch: DataType):
        (
            self.rng,
            self.actor,
            self.critic1,
            self.critic2,
            self.target_critic1_params,
            self.target_critic2_params,
            self.temperature,
            info,
            stats_info,
        ) = _update_jit(
            rng=self.rng,
            batch=batch,
            actor=self.actor,
            critic1=self.critic1,
            critic2=self.critic2,
            target_critic1_params=self.target_critic1_params,
            target_critic2_params=self.target_critic2_params,
            temperature=self.temperature,
            target_entropy=self.target_entropy,
            backup_entropy=self.backup_entropy,
            discount=self.discount,
            tau=self.tau,
        )

        return info, stats_info

@functools.partial(jax.jit, static_argnames="backup_entropy")
def _update_jit(
    *,
    rng: PRNGKey,
    batch: DataType,
    #
    actor: TrainState,
    critic1: TrainState,
    critic2: TrainState,
    target_critic1_params: Params,
    target_critic2_params: Params,
    temperature: TrainState,
    #
    target_entropy: float,
    backup_entropy: bool,
    discount: float,
    tau: float,
):
    # temperature
    temp = temperature()

    # critic target
    rng, key = jax.random.split(rng)
    dist = actor(batch["observations_next"])
    actions_next, log_prob_next = dist.sample_and_log_prob(seed=key)

    state_action_value_next = jnp.minimum(
        critic1.apply_fn({"params": target_critic1_params}, batch["observations_next"], actions_next),
        critic2.apply_fn({"params": target_critic2_params}, batch["observations_next"], actions_next)
    )

    critic_target = batch["rewards"] + (1 - batch["dones"]) * discount * state_action_value_next
    if backup_entropy:
        critic_target -= (1 - batch["dones"]) * discount * temp * log_prob_next

    # critic update
    new_critic1, critic1_info, critic1_stats_info = critic1.update(batch=batch, target=critic_target)
    new_critic2, critic2_info, critic2_stats_info = critic2.update(batch=batch, target=critic_target)

    # actor update
    rng, key = jax.random.split(rng)
    new_actor, actor_info, actor_stats_info = actor.update(batch=batch, critic1=new_critic1, critic2=new_critic2, temp=temp, key=key)

    # temperature update
    new_temperature, temperature_info, temperature_stats_info = temperature.update(entropy=actor_info["entropy"], target_entropy=target_entropy)

    # target_critic update
    new_target_critic1_params = _update_target_net(new_critic1.params, target_critic1_params, tau)
    new_target_critic2_params = _update_target_net(new_critic2.params, target_critic1_params, tau)

    info = {**critic1_info, **critic2_info, **actor_info, **temperature_info}
    stats_info = {**critic1_stats_info, **critic2_stats_info, **actor_stats_info, **temperature_stats_info}
    return (
        rng,
        new_actor,
        new_critic1,
        new_critic2,
        new_target_critic1_params,
        new_target_critic2_params,
        new_temperature,
        info,
        stats_info,
    )

def _actor_loss_fn(
    params: Params,
    state: TrainState,
    batch: DataType,
    critic1: TrainState,
    critic2: TrainState,
    temp: float,
    key: PRNGKey,
) -> Tuple[TrainState, Dict[str, float]]:
    dist = state.apply_fn({"params": params}, batch["observations"], train=True)
    actions, log_prob = dist.sample_and_log_prob(seed=key)

    state_action_value = jnp.minimum(
        critic1(batch["observations"], actions),
        critic2(batch["observations"], actions)
    )

    loss = (log_prob * temp - state_action_value).mean()
    return loss, {f"{state.info_key}_loss": loss, "entropy": -log_prob.mean()}

def _critic_loss_fn(
    params: Params,
    state: TrainState,
    batch: DataType,
    target: jnp.ndarray,
) -> Tuple[TrainState, Dict[str, float]]:
    preds = state.apply_fn({"params": params}, batch["observations"], batch["actions"], train=True)
    loss = ((preds - target)**2).mean()
    return loss, {f"{state.info_key}_loss": loss}

def _temperature_loss_fn(
    params: Params,
    state: TrainState,
    entropy: float,
    target_entropy: float
) -> Tuple[TrainState, Dict[str, float]]:
    temp = state.apply_fn({"params": params})
    loss =  temp * (entropy - target_entropy).mean()
    return loss, {state.info_key: temp, f"{state.info_key}_loss": loss}

def _update_target_net(
    online_params: Params, target_params: Params, tau: int
) -> Tuple[TrainState, Dict[str, float]]:
    new_target_params = jax.tree.map(
        lambda t1, t2: t1 * tau + t2 * (1 - tau),
        online_params, target_params,
    )
    return new_target_params
