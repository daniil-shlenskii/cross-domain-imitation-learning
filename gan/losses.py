import functools
from typing import Callable

from gan.loss_fns import (GradientPenaltyDecorator, d_logistic_loss,
                          d_softplus_loss, g_logistic_loss, g_softplus_loss)

#### gan losses (basic) ####

class GANLoss:
    def __init__(
        self,
        *,
        generator_loss_fn: Callable,
        discriminator_loss_fn: Callable,
        is_generator: bool,
    ):
        if is_generator:
            loss_fn = generator_loss_fn
        else:
            loss_fn = discriminator_loss_fn

        self.loss_fn = loss_fn
        self.generator_loss_fn=generator_loss_fn
        self.discriminator_loss_fn=discriminator_loss_fn
    
    def __call__(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)

LogisticLoss = functools.partial(
    GANLoss,
    generator_loss_fn=g_logistic_loss,
    discriminator_loss_fn=d_logistic_loss,
)

SoftplusLoss = functools.partial(
    GANLoss,
    generator_loss_fn=g_softplus_loss,
    discriminator_loss_fn=d_softplus_loss,
)

class GANLossGP(GANLoss):
    def __init__(
        self,
        *,
        discriminator_loss_fn: Callable,
        gradient_penalty_coef: float = 10.,
        **kwargs,
    ):
        discriminator_loss_fn = GradientPenaltyDecorator(
            d_loss_fn=discriminator_loss_fn,
            gradient_penalty_coef=gradient_penalty_coef
        )

        super().__init__(
            discriminator_loss_fn=discriminator_loss_fn,
            **kwargs,
        )

LogisticLossGP = functools.partial(
    GANLossGP,
    generator_loss_fn=g_logistic_loss,
    discriminator_loss_fn=d_logistic_loss,
)

SoftplusLossGP = functools.partial(
    GANLossGP,
    generator_loss_fn=g_softplus_loss,
    discriminator_loss_fn=d_softplus_loss,
)
