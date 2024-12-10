from copy import deepcopy
from typing import Callable

import jax.numpy as jnp

from gan.discriminator import Discriminator
from gan.losses import GANLoss
from nn.train_state import TrainState
from utils.types import DataType, Params


class DomainEncoderLossMixin:
    def __init__(self, policy_loss: GANLoss, state_loss: GANLoss):
        self._set_policy_loss_fns(policy_loss)
        self._set_state_loss_fns(state_loss)

    def _set_state_loss_fns(self, state_loss: GANLoss):
        """For discriminator deceiving."""
        loss_fn = state_loss.generator_loss_fn
        learner_loss_fn = lambda logits: loss_fn(logits)
        expert_loss_fn = lambda logits: loss_fn(-logits)
        self.learner_state_loss_fn = learner_loss_fn
        self.expert_state_loss_fn = expert_loss_fn

    def _set_policy_loss_fns(self, policy_loss: GANLoss):
        """Helps discriminator to discriminate better."""
        loss_fn = policy_loss.generator_loss_fn
        learner_loss_fn = lambda logits: loss_fn(-logits)
        expert_loss_fn = lambda logits: loss_fn(logits)
        self.learner_policy_loss_fn = learner_loss_fn
        self.expert_policy_loss_fn = expert_loss_fn

    def _loss(
        self,
        params: Params,
        state: TrainState,
        batch: DataType,
        policy_discriminator: Discriminator,
        state_discriminator: Discriminator,
        state_loss_scale: float,
        #
        policy_loss_fn: Callable,
        state_loss_fn: Callable,
    ):
        batch = deepcopy(batch)
        batch["observations"] = state.apply_fn({"params": params}, batch["observations"], train=True)
        batch["observations_next"] = state.apply_fn({"params": params}, batch["observations_next"], train=True)

        policy_batch = jnp.concatenate([batch["observations"], batch["observations_next"]], axis=1)
        policy_logits = policy_discriminator(policy_batch)
        policy_loss = policy_loss_fn(policy_logits).mean()

        state_batch = batch["observations"]
        state_logits = state_discriminator(state_batch)
        state_loss = state_loss_fn(state_logits).mean()

        loss = policy_loss + state_loss_scale * state_loss
        info = {
            f"{state.info_key}_loss": loss,
            f"{state.info_key}_policy_loss": policy_loss,
            f"{state.info_key}_state_loss": state_loss,
            "encoded_batch": batch,
        }
        return loss, info

    def learner_loss(self, *args, **kwargs):
        return self._loss(
            *args,
            **kwargs,
            policy_loss_fn=self.learner_policy_loss_fn,
            state_loss_fn=self.learner_state_loss_fn,
        )

    def expert_loss(self, *args, **kwargs):
        return self._loss(
            *args,
            **kwargs,
            policy_loss_fn=self.expert_policy_loss_fn,
            state_loss_fn=self.expert_state_loss_fn,
        )

class InDomainEncoderLoss(DomainEncoderLossMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        batch: DataType,
        expert_batch: DataType,
        policy_discriminator: Discriminator,
        state_discriminator: Discriminator,
        state_loss_scale: float,
    ):
        learner_loss, learner_info = self.learner_loss(
            params=params,
            state=state,
            batch=batch,
            policy_discriminator=policy_discriminator,
            state_discriminator=state_discriminator,
            state_loss_scale=state_loss_scale,
        )
        expert_loss, expert_info = self.expert_loss(
            params=params,
            state=state,
            batch=expert_batch,
            policy_discriminator=policy_discriminator,
            state_discriminator=state_discriminator,
            state_loss_scale=state_loss_scale,
        )
        loss = (learner_loss + expert_loss) * 0.5
        info = {
            **{"learner_" + k: v for k, v in learner_info.items()},
            **{"expert_" + k: v for k, v in expert_info.items()},
        }
        return loss, info
