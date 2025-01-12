from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable

import jax

from agents.imitation_learning.dida.domain_encoder.utils import project_a_to_b
from gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.types import DataType, Params

from .base import DomainEncoderLossMixin


def remove_projection_a_to_b(a: "PyTree", b: "PyTree"):
    proj = jax.tree.map(project_a_to_b, a, b)
    return jax.tree.map(lambda x, p: x - p, a, proj)



class JaxCustomDomainEncoderProjectedGradLoss(DomainEncoderLossMixin, ABC):
    @classmethod
    def create(cls, project_state_to_policy: bool=True, **kwargs):
        loss = cls(**kwargs)

        jax_custom_loss = jax.custom_vjp(loss, nondiff_argnums=(1, 2, 3, 4,))
        if project_state_to_policy:
            jax_custom_loss.defvjp(loss.forward, loss.backward_state_to_policy)
        else:
            jax_custom_loss.defvjp(loss.forward, loss.backward_policy_to_state)

        return jax_custom_loss

    @abstractmethod
    def forward(self, params: Params, **kwargs):
        pass

    @abstractmethod
    def backward_state_to_policy(self, intermediates, input_grad):
        pass

    @abstractmethod
    def backward_policy_to_state(self, intermediates, input_grad):
        pass

class JaxCustomInDomainEncoderProjectedGradLoss(JaxCustomDomainEncoderProjectedGradLoss):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        print("I AM REALLY IN CALL?")
        pass

    def forward(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        print("custom forward")
        target_random_batch = deepcopy(target_random_batch)
        source_random_batch = deepcopy(source_random_batch)
        source_expert_batch = deepcopy(source_expert_batch)

        # state losses
        (ts_loss, _), ts_grad = jax.value_and_grad(self.target_state_loss, has_aux=True)(
            params,
            state=state,
            discriminator=state_discriminator,
            states=target_random_batch["observations"],
            states_next=target_random_batch["observations_next"],
        )
        (ss_loss, _), ss_grad = jax.value_and_grad(self.source_state_loss, has_aux=True)(
            params,
            state=state,
            discriminator=state_discriminator,
            states=source_expert_batch["observations"],
            states_next=source_expert_batch["observations_next"],
        )

        # policy losses
        (trp_loss, trp_info), trp_grad = jax.value_and_grad(self.target_random_policy_loss, has_aux=True)(
            params,
            state=state,
            discriminator=policy_discriminator,
            states=target_random_batch["observations"],
            states_next=target_random_batch["observations_next"],
        )
        (srp_loss, srp_info), srp_grad = jax.value_and_grad(self.source_random_policy_loss, has_aux=True)(
            params,
            state=state,
            discriminator=policy_discriminator,
            states=source_random_batch["observations"],
            states_next=source_random_batch["observations_next"],
        )
        (sep_loss, sep_info), sep_grad = jax.value_and_grad(self.source_expert_policy_loss, has_aux=True)(
            params,
            state=state,
            discriminator=policy_discriminator,
            states=source_expert_batch["observations"],
            states_next=source_expert_batch["observations_next"],
        )

        # final loss
        state_loss = ts_loss + ss_loss
        policy_loss = trp_loss + srp_loss + sep_loss
        loss = policy_loss + state_loss * self.state_loss_scale

        # update batches
        target_random_batch["observations"] = trp_info["states"]
        target_random_batch["observations_next"] = trp_info["states_next"]
        source_random_batch["observations"] = srp_info["states"]
        source_random_batch["observations_next"] = srp_info["states_next"]
        source_expert_batch["observations"] = sep_info["states"]
        source_expert_batch["observations_next"] = sep_info["states_next"]

        #
        info = {
            f"{state.info_key}/loss": jax.lax.stop_gradient(loss),
            "target_random_batch": jax.lax.stop_gradient(target_random_batch),
            "source_random_batch": jax.lax.stop_gradient(source_random_batch),
            "source_expert_batch": jax.lax.stop_gradient(source_expert_batch)
        }
        intermediates = (
            ts_grad, ss_grad, trp_grad, srp_grad, sep_grad
        )
        print("end")
        return (loss, info), intermediates

    def backward_state_to_policy(
        self,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
        #
        intermediates,
        input_grad
    ):
        print("custom backward")
        input_grad, _ = input_grad
        ts_grad, ss_grad, trp_grad, srp_grad, sep_grad = intermediates

        ts_grad = remove_projection_a_to_b(ts_grad, trp_grad)

        ss_grad = remove_projection_a_to_b(a=ss_grad, b=sep_grad)
        ss_grad = remove_projection_a_to_b(a=ss_grad, b=srp_grad)

        policy_grad = jax.tree.map(lambda x, y, z: z + y + z, trp_grad, srp_grad, sep_grad)
        state_grad = jax.tree.map(lambda x, y: x + y, ts_grad, ss_grad)
        grad = jax.tree.map(lambda x, y: x + y * self.state_loss_scale, policy_grad, state_grad)

        print("end")
        return jax.tree.map(lambda x: x * input_grad, grad),

    def backward_policy_to_state(self, intermediates, input_grad):
        pass



# Domain Encoder Projected Grad Loss

class DomainEncoderProjectedGradLoss(DomainEncoderLossMixin):
    def __init__(self, jax_custom_loss: Callable, **kwargs):
        self.jax_custom_loss = jax_custom_loss
        super().__init__(**kwargs)

    def __call__(self, params: Params, **kwargs):
        loss, info = self.jax_custom_loss(params, **kwargs)
        return loss, info

class InDomainEncoderProjectedGradLoss(DomainEncoderProjectedGradLoss):
    @classmethod
    def create(cls, project_state_to_policy: bool=True, **kwargs):
        jax_custom_loss = JaxCustomInDomainEncoderProjectedGradLoss.create(
            project_state_to_policy=project_state_to_policy,
            **kwargs
        )
        return cls(jax_custom_loss=jax_custom_loss, **kwargs)
