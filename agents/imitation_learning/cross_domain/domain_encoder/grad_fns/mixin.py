from copy import deepcopy

import jax
import jax.numpy as jnp
from typing_extensions import override

from agents.imitation_learning.cross_domain.domain_encoder.losses import \
    DomainEncoderLossMixin
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import Params


class BaseDomainEncoderGradFnMixin(DomainEncoderLossMixin):
    is_grad_fn: bool = True # TODO: temporal crutch

    def target_grad(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        random_batch: jnp.ndarray,
    ):
        # get state and policy grads
        (ts_loss, _), ts_grad =  jax.value_and_grad(self.state_real_loss_given_params, has_aux=True)(
            params,
            state=state,
            states=random_batch["observations"],
            discriminator=state_discriminator,
        )
        (trp_loss, random_states), trp_grad =  jax.value_and_grad(self.policy_fake_loss_given_params, has_aux=True)(
            params,
            state=state,
            states=random_batch["observations"],
            states_next=random_batch["observations_next"],
            discriminator=policy_discriminator,
        )
        random_batch["observations"] = random_states["states"]
        random_batch["observations_next"] = random_states["states_next"]

        # process grads
        ts_grad, trp_grad = self.process_target_grads(
            state_grad=ts_grad, random_policy_grad=trp_grad
        )

        # get resulting grad
        state_grad, policy_grad = ts_grad, trp_grad
        grad = jax.tree.map(lambda x, y: x * self.target_policy_loss_scale + y * self.target_state_loss_scale, policy_grad, state_grad)

        state_loss, policy_loss = ts_loss, trp_loss
        loss = policy_loss * self.target_policy_loss_scale + state_loss * self.target_state_loss_scale

        return grad, loss, {"target_random_batch": random_batch}

    def source_grad(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        random_batch: jnp.ndarray,
        expert_batch: jnp.ndarray,
    ):
        # get state and policy grads
        (ss_loss, _), ss_grad =  jax.value_and_grad(self.state_fake_loss_given_params, has_aux=True)(
            params,
            state=state,
            states=expert_batch["observations"],
            discriminator=state_discriminator,
        )
        (srp_loss, random_states), srp_grad =  jax.value_and_grad(self.policy_fake_loss_given_params, has_aux=True)(
            params,
            state=state,
            states=random_batch["observations"],
            states_next=random_batch["observations_next"],
            discriminator=policy_discriminator,
        )
        random_batch["observations"] = random_states["states"]
        random_batch["observations_next"] = random_states["states_next"]

        (sep_loss, expert_states), sep_grad =  jax.value_and_grad(self.policy_real_loss_given_params, has_aux=True)(
            params,
            state=state,
            states=expert_batch["observations"],
            states_next=expert_batch["observations_next"],
            discriminator=policy_discriminator,
        )
        expert_batch["observations"] = expert_states["states"]
        expert_batch["observations_next"] = expert_states["states_next"]

        # process grads
        ss_grad, srp_grad, sep_grad = self.process_source_grads(
            state_grad=ss_grad, random_policy_grad=srp_grad, expert_policy_grad=sep_grad,
        )

        # get resulting grad
        state_grad = ss_grad
        policy_grad = jax.tree.map(lambda x, y: x + y, sep_grad, srp_grad)
        grad = jax.tree.map(lambda x, y: x * self.source_policy_loss_scale + y * self.source_state_loss_scale, policy_grad, state_grad)

        state_loss, policy_loss = ss_loss, srp_loss + sep_loss
        loss = policy_loss * self.source_policy_loss_scale + state_loss * self.source_state_loss_scale

        return grad, loss, {"source_random_batch": random_batch, "source_expert_batch": expert_batch}

    @override
    def process_target_grads(self, *, state_grad, random_policy_grad):
        return state_grad, random_policy_grad

    @override
    def process_source_grads(self, *, state_grad, random_policy_grad, expert_policy_grad):
        return state_grad, random_policy_grad, expert_policy_grad
