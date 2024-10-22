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

from nn.train_state import TrainState


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]


class SACLearner:
    def __init__(
        self,
        *,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        #
        actor_module_config: DictConfig,
        critic_module_config: DictConfig,
        state_value_fn_module_config: DictConfig,
        #
        actor_params: Params = None,
        critic1_params: Params = None,
        critic2_params: Params = None,
        state_value_fn_params: Params = None,
        #
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        state_value_fn_lr: float = 3e-4,
        #
        discount: float = 0.99,
        tau = 0.005,
    ):
        # reproducability keys
        rng = jax.random.key(seed)
        actor_key, critic1_key, critic2_key, state_value_fn_key = jax.random.split(rng, 4)

        # actor and critic initialization
        observations = observation_space.sample()
        actions = action_space.sample()
        
        actor_module = instantiate(actor_module_config, action_dim=actions.shape)
        critic_module = instantiate(critic_module_config)
        state_value_fn_module = instantiate(state_value_fn_module_config)

        if actor_params is None:
            actor_params = actor_module.init(actor_key, observations, actions)["params"]
        if critic1_params is None:
            critic1_params = critic_module.init(critic1_key, observations, actions)["params"]
        if critic2_params is None:
            critic2_params = critic_module.init(critic2_key, observations, actions)["params"]
        if state_value_fn_params is None:
            state_value_fn_params = state_value_fn_module.init(state_value_fn_key, observations, actions)["params"]

        self.target_state_value_fn_params = state_value_fn_params

        self.actor = TrainState.create(
            loss_fn=self._actor_loss_fn,
            apply_fn=actor_module.apply(),
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr)
        )
        self.critic1 = TrainState.create(
            loss_fn=self._critic_loss_fn,
            apply_fn=critic_module.apply(),
            params=critic1_params,
            tx=optax.adam(learning_rate=critic_lr)
        )
        self.critic2 = TrainState.create(
            loss_fn=self._critic_loss_fn,
            apply_fn=critic_module.apply(),
            params=critic2_params,
            tx=optax.adam(learning_rate=critic_lr)
        )
        self.state_value_fn = TrainState.create(
            loss_fn=self._state_value_fn_loss_fn,
            apply_fn=state_value_fn_module.apply(),
            params=state_value_fn_params,
            tx=optax.adam(learning_rate=state_value_fn_lr)
        )

        #
        self.discount = discount
        self.tau = tau

    def update(self, batch):
        (
            self.actor,
            self.critic,
            self.target_critic_params,
            info,
        ) = self._update_jit(batch)
        return info

    @jax.jit(static_argnames="self")
    def _update(self, batch):
        # state_action_value
        state_action_value = jnp.min(
            critic.apply_fn({"params": critic.params}, batch["obs"], batch["action"])
            for critic in [self.critic1, self.critic2]
        )
        # entropy
        dist = self.actor.apply_fn({"params": self.actor.params}, batch["obs"])
        entropy = -dist.log_prob(batch["obs"]).mean()


        # state_value_fn update
        state_value_target = state_action_value + entropy
        new_state_value_fn = self.state_value_fn.update(state_value_target)

        # critic update
        critic_target = batch["reward"] + self.discount * self.state_value_fn.apply_fn(
            {"params": self.target_state_value_fn_params}, batch["obs_next"]
        )
        new_critic1 = self.critic1.update(critic_target)
        new_critic2 = self.critic2.update(critic_target)

        # actor update
        new_actor = self.actor.update(batch["obs"])

        # target_state_value_fn update
        new_target_state_value_fn_params = (
            new_state_value_fn.params * self.tau +
            self.target_state_value_fn_params * (1 - self.tau)
        )

        info = None

        return (
            new_actor,
            new_critic1,
            new_critic2,
            new_target_state_value_fn_params,
            info,
        )
    
    @staticmethod
    def _state_value_fn_loss_fn(
        params: Params,
        apply_fn: Callable,
        batch: ...,
        target: jnp.ndarray,
        key: PRNGKey,
    ):
        pred = apply_fn({"params": params}, batch["obs"])
        return optax.l2_loss(pred, target)
    
    @staticmethod
    def _critic_loss_fn(
        params: ...,
        apply_fn: TrainState,
        batch: ...,
        target: jnp.ndarray,
        key: PRNGKey,
    ):
        pred = apply_fn({"params": params}, batch["obs"], batch["action"])
        return optax.l2_loss(pred, target)
    
    @staticmethod
    def _actor_loss_fn(
        params: Params,
        apply_fn: Callable,
        batch: ...,
        critic1: TrainState,
        critic2: TrainState,
        key: PRNGKey,
    ):
        dist = apply_fn({"params": params}, batch["obs"])
        action, log_prob = dist.sample_and_log_prob(seed=key)

        state_action_value = jnp.min(
            critic.apply_fn({"params": critic.params}, batch["obs"], action)
            for critic in [critic1, critic2]
        )

        return (log_prob - state_action_value).mean()