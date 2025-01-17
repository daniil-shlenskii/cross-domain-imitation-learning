from copy import deepcopy

import jax
import jax.numpy as jnp
from typing_extensions import override

from agents.imitation_learning.cross_domain.domain_encoder.losses import \
    DomainEncoderLossMixin
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import Params


class DomainEncoderGradFnMixin(DomainEncoderLossMixin):
    is_grad_fn: bool = True # TODO: temporal crutch

    def target_grad(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        random_batch: jnp.ndarray,
    ):
        random_batch = deepcopy(random_batch)

        # get state and policy grads
        (state_loss, _), state_grad =  jax.value_and_grad(self.target_state_loss, has_aux=True)(
            params,
            state=state,
            discriminator=state_discriminator,
            states=random_batch["observations"],
        )
        (random_policy_loss, random_states), random_policy_grad =  jax.value_and_grad(self.target_random_policy_loss, has_aux=True)(
            params,
            state=state,
            discriminator=policy_discriminator,
            states=random_batch["observations"],
            states_next=random_batch["observations_next"],
        )
        random_batch["observations"] = random_states["states"]
        random_batch["observations_next"] = random_states["states_next"]

        # process grads
        state_grad, random_policy_grad = self.process_target_grads(
            state_grad=state_grad, random_policy_grad=random_policy_grad
        )

        # get resulting grad
        policy_grad = random_policy_grad
        grad = jax.tree.map(lambda x, y: x + y * self.state_loss_scale, policy_grad, state_grad)

        policy_loss = random_policy_loss
        loss = policy_loss + state_loss * self.state_loss_scale

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
        random_batch = deepcopy(random_batch)
        expert_batch = deepcopy(expert_batch)

        # get state and policy grads
        (state_loss, _), state_grad =  jax.value_and_grad(self.source_state_loss, has_aux=True)(
            params,
            state=state,
            discriminator=state_discriminator,
            states=expert_batch["observations"],
        )
        (random_policy_loss, random_states), random_policy_grad =  jax.value_and_grad(self.source_random_policy_loss, has_aux=True)(
            params,
            state=state,
            discriminator=policy_discriminator,
            states=random_batch["observations"],
            states_next=random_batch["observations_next"],
        )
        random_batch["observations"] = random_states["states"]
        random_batch["observations_next"] = random_states["states_next"]

        (expert_policy_loss, expert_states), expert_policy_grad =  jax.value_and_grad(self.source_expert_policy_loss, has_aux=True)(
            params,
            state=state,
            discriminator=policy_discriminator,
            states=expert_batch["observations"],
            states_next=expert_batch["observations_next"],
        )
        expert_batch["observations"] = expert_states["states"]
        expert_batch["observations_next"] = expert_states["states_next"]

        # process grads
        state_grad, random_policy_grad, expert_policy_grad = self.process_source_grads(
            state_grad=state_grad, random_policy_grad=random_policy_grad, expert_policy_grad=expert_policy_grad
        )

        # get resulting grad
        policy_grad = jax.tree.map(lambda x, y: x + y, random_policy_grad, expert_policy_grad)
        grad = jax.tree.map(lambda x, y: x + y * self.state_loss_scale, policy_grad, state_grad)

        policy_loss = random_policy_loss + expert_policy_loss
        loss = policy_loss + state_loss * self.state_loss_scale

        return grad, loss, {"source_random_batch": random_batch, "source_expert_batch": expert_batch}

    @override
    def process_target_grads(self, *, state_grad, random_policy_grad):
        return state_grad, random_policy_grad

    @override
    def process_source_grads(self, *, state_grad, random_policy_grad, expert_policy_grad):
        return state_grad, random_policy_grad, expert_policy_grad
