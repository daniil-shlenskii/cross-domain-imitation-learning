from copy import deepcopy

import jax

from agents.imitation_learning.dida.domain_encoder.utils import project_a_to_b
from gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.types import DataType, Params

from .base import DomainEncoderLossMixin


class InDomainEncoderProjectedGradLoss(DomainEncoderLossMixin):
    @classmethod
    def create(cls, *args, **kwargs):
        _loss = cls(*args, **kwargs)
        loss = jax.custom_vjp(_loss)
        loss.defvjp(_loss.forward, _loss.backward)
        return loss

    def __call__(self, *args, **kwargs):
        loss, (info, _) = self.forward(*args, **kwargs)
        return loss, info

    def forward(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        target_random_batch: DataType,
        source_expert_batch: DataType,
        state_loss_scale: float,
    ):
        target_random_batch = deepcopy(target_random_batch)
        source_expert_batch = deepcopy(source_expert_batch)

        # compute state and policy losses
        (ts_loss, _), ts_grad = jax.value_and_grad(self.target_state_loss, has_aux=True)(
            params,
            state=state,
            discriminator=state_discriminator,
            states=target_random_batch["observations"],
        )
        (tp_loss, tp_info), tp_grad = jax.value_and_grad(self.target_policy_loss, has_aux=True)(
            params,
            state=state,
            discriminator=policy_discriminator,
            states=target_random_batch["observations"],
            states_next=target_random_batch["observations_next"],
        )
        (ss_loss, _), ss_grad = jax.value_and_grad(self.source_state_loss, has_aux=True)(
            params,
            state=state,
            discriminator=state_discriminator,
            states=source_expert_batch["observations"],
        )
        (sp_loss, sp_info), sp_grad = jax.value_and_grad(self.source_policy_loss, has_aux=True)(
            params,
            state=state,
            discriminator=policy_discriminator,
            states=source_expert_batch["observations"],
            states_next=source_expert_batch["observations_next"],
        )

        # final loss
        state_loss = ts_loss + ss_loss
        policy_loss = tp_loss + sp_loss
        loss = policy_loss + state_loss * state_loss_scale

        # update batches
        target_random_batch["observations"] = tp_info["states"]
        target_random_batch["observations_next"] = tp_info["states_next"]
        source_expert_batch["observations"] = sp_info["states"]
        source_expert_batch["observations_next"] = sp_info["states_next"]

        #
        info = {
            f"{state.info_key}/loss": loss,
            "target_random_batch": target_random_batch,
            "source_expert_batch": source_expert_batch
        }
        intermediates = (
            ts_grad, tp_grad, ss_grad, sp_grad, state_loss_scale
        )

        return loss, (info, intermediates)

    def backward(self, intermediates, input_grad):
        _, (ts_grad, tp_grad, ss_grad, sp_grad, state_loss_scale) = intermediates
        tp_grad = tp_grad - project_a_to_b(a=tp_grad, b=ts_grad) 
        sp_grad = sp_grad - project_a_to_b(a=sp_grad, b=ss_grad) 
        policy_grad = tp_grad + sp_grad
        state_grad = ts_grad + ss_grad
        grad = policy_grad + state_grad * state_loss_scale
        return grad * input_grad,
