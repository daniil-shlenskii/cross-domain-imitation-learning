from typing import Callable

import jax.numpy as jnp

from gan.discriminator import Discriminator
from gan.losses import GANLoss
from nn.train_state import TrainState
from utils.types import Params


class DomainEncoderLossMixin:
    def __init__(
        self,
        policy_loss: GANLoss,
        state_loss: GANLoss,
        state_loss_scale: float = 1.,
        has_state_discriminator_paired_input: bool = False,
    ):
        self._set_policy_loss_fns(policy_loss)
        self._set_state_loss_fns(state_loss)

        self.state_loss_scale = state_loss_scale

        if has_state_discriminator_paired_input:
            self._state_loss = self._state_paired_input_loss
        else:
            self._state_loss = self._state_single_input_loss

    def _set_state_loss_fns(self, state_loss: GANLoss):
        """For discriminator deceiving."""
        loss_fn = state_loss.generator_loss_fn
        fake_loss_fn = lambda logits: loss_fn(logits)
        real_loss_fn = lambda logits: loss_fn(-logits)
        self.fake_state_loss_fn = fake_loss_fn
        self.real_state_loss_fn = real_loss_fn

    def _set_policy_loss_fns(self, policy_loss: GANLoss):
        """Helps discriminator to discriminate better."""
        loss_fn = policy_loss.generator_loss_fn
        fake_loss_fn = lambda logits: loss_fn(-logits)
        real_loss_fn = lambda logits: loss_fn(logits)
        self.fake_policy_loss_fn = fake_loss_fn
        self.real_policy_loss_fn = real_loss_fn

    # losses templates

    def _state_single_input_loss(
        self,
        params: Params,
        state: TrainState,
        discriminator: Discriminator,
        states: jnp.ndarray,
        states_next: jnp.ndarray,
        #
        state_loss_fn: Callable,
    ):
        states = state.apply_fn({"params": params}, states)
        logits = discriminator(states)
        return state_loss_fn(logits).mean(), {
            "states": states
        }

    def _state_paired_input_loss(
        self,
        params: Params,
        state: TrainState,
        discriminator: Discriminator,
        states: jnp.ndarray,
        states_next: jnp.ndarray,
        #
        state_loss_fn: Callable,
    ):
        states = state.apply_fn({"params": params}, states)
        states_next = state.apply_fn({"params": params}, states_next)
        state_pairs = jnp.concatenate([states, states_next], axis=1)
        logits = discriminator(state_pairs)
        return state_loss_fn(logits).mean(), {
            "states": states,
            "states_next": states_next,
        }

    def _policy_loss(
        self,
        params: Params,
        state: TrainState,
        discriminator: Discriminator,
        states: jnp.ndarray,
        states_next: jnp.ndarray,
        #
        policy_loss_fn: Callable,
    ):
        states = state.apply_fn({"params": params}, states)
        states_next = state.apply_fn({"params": params}, states_next)
        state_pairs = jnp.concatenate([states, states_next], axis=1)
        logits = discriminator(state_pairs)
        return policy_loss_fn(logits).mean(), {
            "states": states,
            "states_next": states_next,
        }

    # losses

    def target_state_loss(self, *args, **kwargs):
        return self._state_loss(
            *args,
            **kwargs,
            state_loss_fn=self.fake_state_loss_fn,
        )

    def source_state_loss(self, *args, **kwargs):
        return self._state_loss(
            *args,
            **kwargs,
            state_loss_fn=self.real_state_loss_fn,
        )

    def target_random_policy_loss(self, *args, **kwargs):
        return self._policy_loss(
            *args,
            **kwargs,
            policy_loss_fn=self.fake_policy_loss_fn,
        )

    def source_random_policy_loss(self, *args, **kwargs):
        return self._policy_loss(
            *args,
            **kwargs,
            policy_loss_fn=self.fake_policy_loss_fn,
        )

    def source_expert_policy_loss(self, *args, **kwargs):
        return self._policy_loss(
            *args,
            **kwargs,
            policy_loss_fn=self.real_policy_loss_fn,
        )
