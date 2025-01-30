from typing import Callable

import jax.numpy as jnp

from misc.gan.discriminator import Discriminator
from misc.gan.losses import GANLoss
from nn.train_state import TrainState
from utils.custom_types import Params


class DomainEncoderLossMixin:
    def __init__(
        self,
        policy_loss: GANLoss,
        state_loss: GANLoss,
        target_state_loss_scale: float = 1.,
        target_policy_loss_scale: float = 1.,
        source_state_loss_scale: float = 1.,
        source_policy_loss_scale: float = 1.,
        state_loss_scale: float = None,
        policy_loss_scale: float = None,
        update_target_encoder_with_policy_discrminator: bool = True,
    ):
        self.state_fake_loss_fn, self.state_real_loss_fn = self._get_fake_and_real_generator_loss_fns(state_loss)
        self.policy_fake_loss_fn, self.policy_real_loss_fn = self._get_fake_and_real_generator_loss_fns(policy_loss)

        self.target_state_loss_scale = target_state_loss_scale if state_loss_scale is None else state_loss_scale
        self.target_policy_loss_scale = target_policy_loss_scale if policy_loss_scale is None else policy_loss_scale
        self.source_state_loss_scale = source_state_loss_scale if state_loss_scale is None else state_loss_scale
        self.source_policy_loss_scale = source_policy_loss_scale if policy_loss_scale is None else policy_loss_scale
        self.update_target_encoder_with_policy_discrminator = float(update_target_encoder_with_policy_discrminator)

    def _get_fake_and_real_generator_loss_fns(self, loss: GANLoss):
        loss_fn = loss.generator_loss_fn
        fake_loss_fn = lambda logits: loss_fn(-logits)
        real_loss_fn = lambda logits: loss_fn(logits)
        return fake_loss_fn, real_loss_fn

    # losses templates

    def _state_loss(
        self,
        *,
        states: jnp.ndarray,
        discriminator: Discriminator,
        state_loss_fn: Callable,
    ):
        logits = discriminator(states)
        return state_loss_fn(logits).mean()

    def _policy_loss(
        self,
        *,
        states: jnp.ndarray,
        states_next: jnp.ndarray,
        discriminator: Discriminator,
        policy_loss_fn: Callable,
    ):
        state_pairs = jnp.concatenate([states, states_next], axis=1)
        logits = discriminator(state_pairs)
        return policy_loss_fn(logits).mean()

   # losses

    def state_fake_loss(self, *args, **kwargs):
        return self._state_loss(
            *args,
            **kwargs,
            state_loss_fn=self.state_fake_loss_fn,
        )

    def state_real_loss(self, *args, **kwargs):
        return self._state_loss(
            *args,
            **kwargs,
            state_loss_fn=self.state_real_loss_fn,
        )

    def policy_fake_loss(self, *args, **kwargs):
        return self._policy_loss(
            *args,
            **kwargs,
            policy_loss_fn=self.policy_fake_loss_fn,
        )

    def policy_real_loss(self, *args, **kwargs):
        return self._policy_loss(
            *args,
            **kwargs,
            policy_loss_fn=self.policy_real_loss_fn,
        )
