from typing import Any, Callable

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
        value_critic_module_config: DictConfig,
        #
        actor_params: Params = None,
        critic1_params: Params = None,
        critic2_params: Params = None,
        value_critic_params: Params = None,
        #
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_critic_lr: float = 3e-4,
        #
        discount: float = 0.99,
        tau = 0.005,
    ):
        # reproducability keys
        rng = jax.random.key(seed)
        rng, actor_key, critic1_key, critic2_key, value_critic_key = jax.random.split(rng, 5)
        self._rng = rng

        # actor and critic initializationО
        observation = observation_space.sample()
        action = action_space.sample()

        actor_module = instantiate(actor_module_config, action_dim=action.shape[-1])
        critic_module = instantiate(critic_module_config)
        value_critic_module = instantiate(value_critic_module_config)

        if actor_params is None:
            actor_params = actor_module.init(actor_key, observation)["params"]
        if critic1_params is None:
            critic1_params = critic_module.init(critic1_key, observation, action)["params"]
        if critic2_params is None:
            critic2_params = critic_module.init(critic2_key, observation, action)["params"]
        if value_critic_params is None:
            value_critic_params = value_critic_module.init(value_critic_key, observation)["params"]

        self.target_value_critic_params = value_critic_params

        self.actor = TrainState.create(
            loss_fn=self._actor_loss_fn,
            apply_fn=actor_module.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr)
        )
        self.critic1 = TrainState.create(
            loss_fn=self._critic_loss_fn,
            apply_fn=critic_module.apply,
            params=critic1_params,
            tx=optax.adam(learning_rate=critic_lr)
        )
        self.critic2 = TrainState.create(
            loss_fn=self._critic_loss_fn,
            apply_fn=critic_module.apply,
            params=critic2_params,
            tx=optax.adam(learning_rate=critic_lr)
        )
        self.value_critic = TrainState.create(
            loss_fn=self._value_critic_loss_fn,
            apply_fn=value_critic_module.apply,
            params=value_critic_params,
            tx=optax.adam(learning_rate=value_critic_lr)
        )

        #
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
            self.target_value_critic_params,
            info,
        ) = self._update_jit(batch, rng=self._rng)
        return info

    @functools.partial(jax.jit, static_argnames="self")
    def _update_jit(self, batch, rng: PRNGKey):
        # reproducibility
        rng, key = jax.random.split(rng)

        # state_action_value
        state_action_value = jnp.min(
            jnp.stack([
                critic.apply_fn({"params": critic.params}, batch["observations"], batch["actions"])
                for critic in [self.critic1, self.critic2]
            ]),
            axis=0,
        )
        # log_prob
        dist = self.actor.apply_fn({"params": self.actor.params}, batch["observations"])
        log_prob = dist.log_prob(batch["actions"]).mean()

        # value_critic update
        state_value_target = state_action_value - log_prob
        new_value_critic, value_critic_info = self.value_critic.update(
            batch=batch, target=state_value_target
        )

        # target_value_critic update
        new_target_value_critic_params = jax.tree.map(
            lambda t1, t2: t1 * self.tau + t2 * (1 - self.tau),
            new_value_critic.params,
            self.target_value_critic_params,
        )

        # critic update
        critic_target = batch["rewards"] + self.discount * self.value_critic.apply_fn(
            {"params": new_target_value_critic_params}, batch["observations_next"]
        )
        new_critic1, critic1_info = self.critic1.update(
            batch=batch, target=critic_target, critic_idx=1
        )
        new_critic2, critic2_info = self.critic2.update(
            batch=batch, target=critic_target, critic_idx=2
        )

        # actor update
        new_actor, actor_info = self.actor.update(
            batch=batch, critic1=new_critic1, critic2=new_critic2, key=key
        )

        info = {**value_critic_info, **critic1_info, **critic2_info, **actor_info}
        return (
            rng,
            new_actor,
            new_critic1,
            new_critic2,
            new_target_value_critic_params,
            info,
        )
    
    @staticmethod
    def _value_critic_loss_fn(
        params: Params,
        apply_fn: Callable,
        batch: DataType,
        target: jnp.ndarray,
    ):
        pred = apply_fn({"params": params}, batch["observations"])
        loss = optax.l2_loss(pred, target).mean()
        return loss, {"value_critic_loss": loss}
    
    @staticmethod
    def _critic_loss_fn(
        params: Params,
        apply_fn: TrainState,
        batch: DataType,
        target: jnp.ndarray,
        critic_idx: int,
    ):
        pred = apply_fn({"params": params}, batch["observations"], batch["actions"])
        loss = optax.l2_loss(pred, target).mean()
        return loss, {f"critic{critic_idx}_loss": loss}
    
    @staticmethod
    def _actor_loss_fn(
        params: Params,
        apply_fn: Callable,
        batch: DataType,
        critic1: TrainState,
        critic2: TrainState,
        key: PRNGKey,
    ):
        dist = apply_fn({"params": params}, batch["observations"])
        actions = dist.sample(seed=key)
        log_prob = dist.log_prob(actions)

        state_action_value = jnp.min(
            jnp.stack([
                critic.apply_fn({"params": critic.params}, batch["observations"], actions)
                for critic in [critic1, critic2]
            ]),
            axis=0
        )

        loss = (log_prob - state_action_value).mean()
        return loss, {"actor_loss": loss, "entropy": -log_prob.mean()}
