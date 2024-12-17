from copy import deepcopy
from typing import Callable

import jax.numpy as jnp

from agents.imitation_learning.utils import get_state_pairs
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

    def target_state_loss_fn(
        self,
        discriminator: Discriminator,
        random_state_pairs: jnp.ndarray,
        expert_state_pairs: jnp.ndarray,
    ):
        state_pairs = jnp.concatenate([random_state_pairs, expert_state_pairs])
        state_pairs_logits = discriminator(state_pairs)
        return self.fake_state_loss_fn(state_pairs_logits).mean()

    def target_policy_loss_fn(
        self,
        discriminator: Discriminator,
        random_state_pairs: jnp.ndarray,
        expert_state_pairs: jnp.ndarray,
    ):
        state_pairs_logits = discriminator(random_state_pairs)
        return self.fake_policy_loss_fn(state_pairs_logits).mean()

    def source_state_loss_fn(
        self,
        discriminator: Discriminator,
        random_state_pairs: jnp.ndarray,
        expert_state_pairs: jnp.ndarray,
    ):
        state_pairs = jnp.concatenate([random_state_pairs, expert_state_pairs])
        state_pairs_logits = discriminator(state_pairs)
        return self.real_state_loss_fn(state_pairs_logits).mean()

    def source_policy_loss_fn(
        self,
        discriminator: Discriminator,
        random_state_pairs: jnp.ndarray,
        expert_state_pairs: jnp.ndarray,
    ):
        random_state_pairs_logits = discriminator(random_state_pairs)
        random_loss = self.fake_policy_loss_fn(random_state_pairs_logits).mean()

        expert_state_pairs_logits = discriminator(expert_state_pairs)
        expert_loss = self.real_policy_loss_fn(expert_state_pairs_logits).mean()

        return (random_loss + expert_loss) * 0.5

    def _loss(
        self,
        params: Params,
        state: TrainState,
        random_batch: DataType,
        expert_batch: DataType,
        policy_discriminator: Discriminator,
        state_discriminator: Discriminator,
        state_loss_scale: float,
        #
        state_loss_fn: Callable,
        policy_loss_fn: Callable,
    ):
        # prepre state pairs
        random_batch = deepcopy(random_batch)
        expert_batch = deepcopy(expert_batch)
        for k in ["observations", "observations_next"]:
            random_batch[k] = state.apply_fn({"params": params}, random_batch[k], train=True)
            expert_batch[k] = state.apply_fn({"params": params}, expert_batch[k], train=True)
        random_state_pairs = get_state_pairs(random_batch)
        expert_state_pairs = get_state_pairs(expert_batch)

        # compute state loss
        state_loss = state_loss_fn(
            discriminator=state_discriminator,
            random_state_pairs=random_state_pairs,
            expert_state_pairs=expert_state_pairs,
        )

        # compute policy loss
        policy_loss = policy_loss_fn(
            discriminator=policy_discriminator,
            random_state_pairs=random_state_pairs,
            expert_state_pairs=expert_state_pairs,
        )

        loss = policy_loss + state_loss_scale * state_loss
        return loss, policy_loss, state_loss, random_batch, expert_batch

    def target_loss(self, *args, **kwargs):
        return self._loss(
            *args,
            **kwargs,
            state_loss_fn=self.target_state_loss_fn,
            policy_loss_fn=self.target_policy_loss_fn,
        )

    def source_loss(self, *args, **kwargs):
        return self._loss(
            *args,
            **kwargs,
            state_loss_fn=self.source_state_loss_fn,
            policy_loss_fn=self.source_policy_loss_fn,
        )

class InDomainEncoderLoss(DomainEncoderLossMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        target_random_batch: DataType,
        target_expert_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
        policy_discriminator: Discriminator,
        state_discriminator: Discriminator,
        state_loss_scale: float,
    ):
        (
            target_loss,
            target_policy_loss,
            target_state_loss,
            target_random_batch,
            target_expert_batch,
        ) = self.target_loss(
            params=params,
            state=state,
            random_batch=target_random_batch,
            expert_batch=target_expert_batch,
            policy_discriminator=policy_discriminator,
            state_discriminator=state_discriminator,
            state_loss_scale=state_loss_scale,
        )
        (
            source_loss,
            source_policy_loss,
            source_state_loss,
            source_random_batch,
            source_expert_batch,
        ) = self.source_loss(
            params=params,
            state=state,
            random_batch=source_random_batch,
            expert_batch=source_expert_batch,
            policy_discriminator=policy_discriminator,
            state_discriminator=state_discriminator,
            state_loss_scale=state_loss_scale,
        )
        loss = (target_loss + source_loss) * 0.5
        info = {
            state.info_key: loss,
            f"{state.info_key}/target_policy_loss": target_policy_loss,
            f"{state.info_key}/target_state_loss": target_state_loss,
            f"{state.info_key}/source_policy_loss": source_policy_loss,
            f"{state.info_key}/source_state_loss": source_state_loss,
            "target_random_batch": target_random_batch,
            "target_expert_batch": target_expert_batch,
            "source_random_batch": source_random_batch,
            "source_expert_batch": source_expert_batch,           
        }
        return loss, info

class CrossDomainTargetEncoderLoss(DomainEncoderLossMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        target_random_batch: DataType,
        target_expert_batch: DataType,
        policy_discriminator: Discriminator,
        state_discriminator: Discriminator,
        state_loss_scale: float,
    ):
        (
            target_loss,
            target_policy_loss,
            target_state_loss,
            target_random_batch,
            target_expert_batch,
        ) = self.target_loss(
            params=params,
            state=state,
            random_batch=target_random_batch,
            expert_batch=target_expert_batch,
            policy_discriminator=policy_discriminator,
            state_discriminator=state_discriminator,
            state_loss_scale=state_loss_scale,
        )
        info = {
            state.info_key: target_loss,
            f"{state.info_key}/target_policy_loss": target_policy_loss,
            f"{state.info_key}/target_state_loss": target_state_loss,
            "target_random_batch": target_random_batch,
            "target_expert_batch": target_expert_batch,
        }
        return target_loss, info

class CrossDomainSourceEncoderLoss(DomainEncoderLossMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        source_random_batch: DataType,
        source_expert_batch: DataType,
        policy_discriminator: Discriminator,
        state_discriminator: Discriminator,
        state_loss_scale: float,
    ):
        (
            source_loss,
            source_policy_loss,
            source_state_loss,
            source_random_batch,
            source_expert_batch,
        ) = self.source_loss(
            params=params,
            state=state,
            random_batch=source_random_batch,
            expert_batch=source_expert_batch,
            policy_discriminator=policy_discriminator,
            state_discriminator=state_discriminator,
            state_loss_scale=state_loss_scale,
        )
        info = {
            state.info_key: source_loss,
            f"{state.info_key}/source_policy_loss": source_policy_loss,
            f"{state.info_key}/source_state_loss": source_state_loss,
            "source_random_batch": source_random_batch,
            "source_expert_batch": source_expert_batch,
        }
        return source_loss, info
