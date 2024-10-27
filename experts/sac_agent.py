from typing import Any, Callable, Dict, Tuple
from copy import deepcopy

import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn

import numpy as np
import gymnasium as gym

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import functools

from nn.train_state import TrainState

from experts.base_agent import Agent
from utils.types import DataType, Params, PRNGKey
from utils.reproducibility import seed_by_key


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
        critic1_params: Params = None,
        critic2_params: Params = None,
        temperature_params: Params = None,
        #
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temperature_lr: float = 3e-4,
        #
        target_entropy: float = None,
        backup_entropy: bool = False,
        discount: float = 0.99,
        tau = 0.005,
    ):
        # reproducability keys
        rng = jax.random.key(seed)
        rng, actor_key, critic1_key, critic2_key, temperature_key = jax.random.split(rng, 5)
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
        if critic1_params is None:
            critic1_params = critic_module.init(critic1_key, observation, action)["params"]
        if critic2_params is None:
            critic2_params = critic_module.init(critic2_key, observation, action)["params"]
        if temperature_params is None:
            temperature_params = temperature_module.init(temperature_key)["params"]

        self.target_critic1_params = deepcopy(critic1_params)
        self.target_critic2_params = deepcopy(critic2_params)

        self.actor = TrainState.create(
            loss_fn=_actor_loss_fn,
            apply_fn=actor_module.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr)
        )
        self.critic1 = TrainState.create(
            loss_fn=_critic_loss_fn,
            apply_fn=critic_module.apply,
            params=critic1_params,
            tx=optax.adam(learning_rate=critic_lr)
        )
        self.critic2 = TrainState.create(
            loss_fn=_critic_loss_fn,
            apply_fn=critic_module.apply,
            params=critic2_params,
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
        self.backup_entropy = backup_entropy
        self.discount = discount
        self.tau = tau

    @property
    def critic(self) -> TrainState:
        return self.critic1

    def update(self, batch: DataType):
        (
            self._rng,
            self.actor,
            self.critic1,
            self.critic2,
            self.temperature,
            self.new_target_critic1_params,
            self.new_target_critic2_params,
            info,
        ) = _update_jit(
            batch,
            rng=self._rng,
            actor=self.actor,
            critic1=self.critic1,
            critic2=self.critic2,
            temperature=self.temperature,
            target_critic1_params=self.target_critic1_params,
            target_critic2_params=self.target_critic2_params,
            target_entropy=self.target_entropy,
            backup_entropy=self.backup_entropy,
            discount=self.discount,
            tau=self.tau,
        )
        return info

@functools.partial(jax.jit, static_argnames="backup_entropy")
def _update_jit(
    batch,
    rng: PRNGKey,
    #
    actor: TrainState,
    critic1: TrainState,
    critic2: TrainState,
    temperature: TrainState,
    target_critic1_params: Params,
    target_critic2_params: Params,
    #
    target_entropy: float,
    backup_entropy: bool,
    discount: float,
    tau: float,
):
    rng, key = jax.random.split(rng)

    # temperature
    temp = temperature()

    # log_prob
    dist = actor(batch["observations_next"])
    actions_next = dist.sample(seed=key)
    log_prob_next = dist.log_prob(actions_next)

    # state-action values
    state_action_value1_next = critic1(batch["observations_next"], actions_next)
    state_action_value2_next = critic2(batch["observations_next"], actions_next)

    # critic target
    state_action_value_next = jnp.min(
        jnp.stack([state_action_value1_next, state_action_value2_next]),
        axis=0
    )
    state_value_next = state_action_value_next - log_prob_next
    critic_target = batch["rewards"] + (1 - batch["dones"]) * discount * state_value_next
    if backup_entropy:
        critic_target -= (1 - batch["dones"]) * discount * temp * log_prob_next

    # critic update
    new_critic1, critic1_info = critic1.update(batch=batch, target=critic_target, critic_idx=1)
    new_critic2, critic2_info = critic2.update(batch=batch, target=critic_target, critic_idx=2)

    # actor update
    new_actor, actor_info = actor.update(batch=batch, critic1=new_critic1, critic2=new_critic2, temp=temp, key=key)

    # temperature update
    new_temperature, temperature_info = temperature.update(entropy=actor_info["entropy"], target_entropy=target_entropy)

    # target_critic update
    new_target_critic1_params = _update_target_net(new_critic1.params, target_critic1_params, tau)
    new_target_critic2_params = _update_target_net(new_critic2.params, target_critic2_params, tau)

    info = {**critic1_info, **critic2_info, **actor_info, **temperature_info}
    return (
        rng,
        new_actor,
        new_critic1,
        new_critic2,
        new_temperature,
        new_target_critic1_params,
        new_target_critic2_params,
        info,
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
    dist = state.apply_fn({"params": params}, batch["observations"])
    actions = dist.sample(seed=key)
    log_prob = dist.log_prob(actions)

    state_action_value = jnp.min(
        jnp.stack([
            critic.apply_fn({"params": critic.params}, batch["observations"], actions)
            for critic in [critic1, critic2]
        ]),
        axis=0
    )

    loss = (log_prob * temp - state_action_value).mean()
    return loss, {"actor_loss": loss, "entropy": -log_prob.mean()}

def _critic_loss_fn(
    params: Params,
    state: TrainState,
    batch: DataType,
    target: jnp.ndarray,
    critic_idx: int,
) -> Tuple[TrainState, Dict[str, float]]:
    pred = state.apply_fn({"params": params}, batch["observations"], batch["actions"])
    loss = optax.l2_loss(pred, target).mean()
    return loss, {f"critic{critic_idx}_loss": loss}

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
