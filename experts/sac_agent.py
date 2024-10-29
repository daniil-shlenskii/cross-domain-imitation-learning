import functools
from copy import deepcopy
from typing import Dict, Tuple

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import instantiate
from nn.train_state import TrainState
from omegaconf.dictconfig import DictConfig
from utils.types import DataType, Params, PRNGKey

from experts.base_agent import Agent


class SACAgent(Agent):
    def __init__(
        self,
        *,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        #
        actor_module_config: DictConfig,
        critic_module_config: DictConfig,
        temperature_module_config: DictConfig,
        #
        actor_params: Params = None,
        critic_params: Params = None,
        temperature_params: Params = None,
        #
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temperature_lr: float = 3e-4,
        #
        update_temperature: bool = True,
        target_entropy: float = None,
        backup_entropy: bool = True,
        #
        discount: float = 0.99,
        tau = 0.005,
    ):
        # reproducibility keys
        rng = jax.random.key(seed)
        rng, actor_key, critic_key, temperature_key = jax.random.split(rng, 4)
        self._rng = rng

        # actor and critic initialization
        observation = observation_space.sample()
        action = action_space.sample()
        action_dim = action.shape[-1]

        actor_module = instantiate(actor_module_config, action_dim=action_dim)
        critic_module = instantiate(critic_module_config)
        temperature_module = instantiate(temperature_module_config)

        if actor_params is None:
            actor_params = actor_module.init(actor_key, observation)["params"]
        if critic_params is None:
            critic_params = critic_module.init(critic_key, observation, action)["params"]
        if temperature_params is None:
            temperature_params = temperature_module.init(temperature_key)["params"]

        self.target_critic_params = deepcopy(critic_params)

        self.actor = TrainState.create(
            loss_fn=_actor_loss_fn,
            apply_fn=actor_module.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr)
        )
        self.critic = TrainState.create(
            loss_fn=_critic_loss_fn,
            apply_fn=critic_module.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr)
        )
        self.temperature = TrainState.create(
            loss_fn=_temperature_loss_fn,
            apply_fn=temperature_module.apply,
            params=temperature_params,
            tx=optax.adam(learning_rate=temperature_lr)
        )

        # target entropy init
        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        #
        self.seed = seed
        self.update_temperature = update_temperature
        self.backup_entropy = backup_entropy
        self.discount = discount
        self.tau = tau

    def update(self, batch: DataType):
        (
            self._rng,
            self.actor,
            self.critic,
            new_temperature,
            self.target_critic_params,
            info,
        ) = _update_jit(
            batch,
            rng=self._rng,
            actor=self.actor,
            critic=self.critic,
            temperature=self.temperature,
            target_critic_params=self.target_critic_params,
            target_entropy=self.target_entropy,
            backup_entropy=self.backup_entropy,
            discount=self.discount,
            tau=self.tau,
        )

        if self.update_temperature:
            self.temperature = new_temperature

        return info

@functools.partial(jax.jit, static_argnames="backup_entropy")
def _update_jit(
    batch: DataType,
    rng: PRNGKey,
    #
    actor: TrainState,
    critic: TrainState,
    temperature: TrainState,
    target_critic_params: Params,
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
    actions_next = dist.sample(seed=key)
    log_prob_next = dist.log_prob(actions_next)

    state_action_value_next = critic.apply_fn({"params": target_critic_params}, batch["observations_next"], actions_next)

    critic_target = batch["rewards"] + (1 - batch["dones"]) * discount * state_action_value_next
    if backup_entropy:
        critic_target -= (1 - batch["dones"]) * discount * temp * log_prob_next

    # critic update
    new_critic, critic_info = critic.update(batch=batch, target=critic_target)

    # actor update
    rng, key = jax.random.split(rng)
    new_actor, actor_info = actor.update(batch=batch, critic=new_critic, temp=temp, key=key)

    # temperature update
    new_temperature, temperature_info = temperature.update(entropy=actor_info["entropy"], target_entropy=target_entropy)

    # target_critic update
    new_target_critic_params = _update_target_net(new_critic.params, target_critic_params, tau)

    info = {**critic_info, **actor_info, **temperature_info}
    return (
        rng,
        new_actor,
        new_critic,
        new_temperature,
        new_target_critic_params,
        info,
    )

def _actor_loss_fn(
    params: Params,
    state: TrainState,
    batch: DataType,
    critic: TrainState,
    temp: float,
    key: PRNGKey,
) -> Tuple[TrainState, Dict[str, float]]:
    dist = state.apply_fn({"params": params}, batch["observations"])
    actions = dist.sample(seed=key)
    log_prob = dist.log_prob(actions)

    state_action_value = critic.apply_fn({"params": critic.params}, batch["observations"], actions)

    loss = (log_prob * temp - state_action_value).mean()
    return loss, {"actor_loss": loss, "entropy": -log_prob.mean()}

def _critic_loss_fn(
    params: Params,
    state: TrainState,
    batch: DataType,
    target: jnp.ndarray,
) -> Tuple[TrainState, Dict[str, float]]:
    pred = state.apply_fn({"params": params}, batch["observations"], batch["actions"])
    loss = optax.l2_loss(pred, target).mean()
    return loss, {f"critic_loss": loss}

def _temperature_loss_fn(
    params: Params,
    state: TrainState,
    entropy: float,
    target_entropy: float
) -> Tuple[TrainState, Dict[str, float]]:
    temp = state.apply_fn({"params": params})
    loss =  temp * (entropy - target_entropy).mean()
    return loss, {"temperature": temp, "temperature_loss": loss}

def _update_target_net(
    online_params: Params, target_params: Params, tau: int
) -> Tuple[TrainState, Dict[str, float]]:
    new_target_params = jax.tree.map(
        lambda t1, t2: t1 * tau + t2 * (1 - tau),
        online_params, target_params,
    )
    return new_target_params
