import functools
from typing import Callable

import jax
import jax.numpy as jnp

from gan.base_losses import (d_logistic_loss, d_softplus_loss,
                             g_nonsaturating_logistic_loss,
                             g_nonsaturating_softplus_loss, gradient_penalty)
from gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.types import Params

#### gan losses (basic) ####

class GANLoss:
    def __init__(
        self,
        *,
        generator_loss_fn: Callable,
        discriminator_loss_fn: Callable,
        is_generator: bool,
    ):
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn

        if is_generator:
            loss_fn = self.generator_loss
        else:
            loss_fn = self.discriminator_loss

        self.loss_fn = loss_fn

    def generator_loss(
        self,
        params: Params,
        state: TrainState,
        batch: jnp.ndarray,
        discriminator: Discriminator,
        train: bool = True,
    ):
        fake_batch = state.apply_fn({"params": params}, batch, train=train)
        fake_logits = discriminator(fake_batch)
        loss = self.generator_loss_fn(fake_logits)

        info = {
            f"{state.info_key}_loss": loss,
            "generations": fake_batch
        }
        return loss, info

    def discriminator_loss(
        self,
        params: Params,
        state: TrainState,
        real_batch: jnp.ndarray,
        fake_batch: jnp.ndarray,
        train: bool = True,
    ):
        real_logits = state.apply_fn({"params": params}, real_batch, train=train)
        fake_logits = state.apply_fn({"params": params}, fake_batch, train=train)
        loss = self.discriminator_loss_fn(real_logits=real_logits, fake_logits=fake_logits)

        info = {
            f"{state.info_key}_loss": loss,
            "real_logits": real_logits,
            "fake_logits": fake_logits,
        }
        return loss, info

    def __call__(
        self,
        params: Params,
        state: TrainState,
        **kwargs,
    ):
        return self.loss_fn(
            params=params,
            state=state,
            **kwargs
        )

class LogisticLoss(GANLoss):
    def __init__(self, is_generator):
        super().__init__(
            generator_loss_fn=g_nonsaturating_logistic_loss,
            discriminator_loss_fn=d_logistic_loss,
            is_generator=is_generator,
        )

class SoftplusLoss(GANLoss):
    def __init__(self, is_generator):
        super().__init__(
            generator_loss_fn=g_nonsaturating_softplus_loss,
            discriminator_loss_fn=d_softplus_loss,
            is_generator=is_generator,
        )

#### gan losses with gradient penalty ####

class GradientPenaltyDecorator:
    def __init__(
        self,
        d_loss_fn: Callable,
        gradient_penalty_coef: float
    ):
        self.d_loss_fn = d_loss_fn
        self.penalty_coef = gradient_penalty_coef

    def __call__(
        self,
        params: Params,
        state: TrainState,
        real_batch: jnp.ndarray,
        fake_batch: jnp.ndarray,
        **kwargs,
    ):
        d_loss, info = self.d_loss_fn(
            params=params,
            state=state,
            real_batch=real_batch,
            fake_batch=fake_batch,
            **kwargs
        )

        disc_grad_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x, train=True))
        penalty = gradient_penalty(key=jax.random.key(state.step), real_batch=real_batch, fake_batch=fake_batch, discriminator_grad_fn=disc_grad_fn)

        loss_with_gp = d_loss + self.penalty_coef * penalty

        info.update({
            f"{state.info_key}_loss_with_gradient_penalty": loss_with_gp,
            f"{state.info_key}_gradient_penalty": penalty
        })

        return loss_with_gp, info

def _gan_loss_with_gradient_penalty_decorator(cls: GANLoss):
    class GANLossGP(cls):
        def __init__(self, *, gradient_penalty_coef: float, **kwargs):
            self.discriminator_loss = GradientPenaltyDecorator(
                d_loss_fn=self.discriminator_loss,
                gradient_penalty_coef=gradient_penalty_coef
            )
            super().__init__(**kwargs)
    return GANLossGP

LogisticLossGP = _gan_loss_with_gradient_penalty_decorator(LogisticLoss)

SoftplusLossGP = _gan_loss_with_gradient_penalty_decorator(SoftplusLoss)
