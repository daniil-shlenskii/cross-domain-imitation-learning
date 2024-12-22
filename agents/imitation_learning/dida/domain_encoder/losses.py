from copy import deepcopy
from typing import Callable

import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig

from gan.discriminator import Discriminator
from gan.losses import GANLoss
from nn.train_state import TrainState
from utils.types import DataType, Params


class DomainEncoderLossMixin:
    def __init__(
        self,
        policy_loss: GANLoss,
        state_loss: GANLoss,
        grads_processor_config: DictConfig=None,
    ):
        self._set_policy_loss_fns(policy_loss)
        self._set_state_loss_fns(state_loss)
        if grads_processor_config is None:
            def grads_processor(state_grad, policy_grad, state_loss_scale):
                return policy_grad + state_grad * state_loss_scale, {}
        else:
            grads_processor = instantiate(grads_processor_config)
        self.grads_processor = grads_processor


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

    def _state_loss(
        self,
        params: Params,
        state: TrainState,
        discriminator: Discriminator,
        states: jnp.ndarray,
        #
        state_loss_fn: Callable,
    ):
        states = state.apply_fn({"params": params}, states)
        logits = discriminator(states)
        return state_loss_fn(logits).mean(), {
            "states": states
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

    # # grad function template
    #
    # def _grad_fn(
    #     params: Params,
    #     state: TrainState,
    #     state_discriminator: Discriminator,
    #     policy_discriminator: Discriminator,
    #     states: jnp.ndarray,
    #     states_next: jnp.ndarray,
    #     state_loss_scale: float,
    #     #
    #     info_key: str,
    #     state_loss: Callable,
    #     policy_loss: Callable,
    #     grads_processor: Callable,
    # ):
    #     (state_loss, _), state_grad = jax.vmap(jax.value_and_grad(state_loss, has_aux=True))(
    #         params,
    #         state=state,
    #         discriminator=state_discriminator,
    #         state=states,
    #     )
    #     (policy_loss, policy_info), policy_grad = jax.vmap(jax.value_and_grad(policy_loss, has_aux=True))(
    #         params,
    #         state=state,
    #         discriminator=policy_discriminator,
    #         states=states,
    #         states_next=states_next,
    #     )
    #     grad, info = grads_processor(state_grad, policy_grad, state_loss_scale)
    #
    #     # prepare info
    #     info["state_loss"] = state_loss
    #     info["policy_loss"] = policy_loss
    #     info["loss"] = policy_loss + state_loss * state_loss_scale
    #
    #     ## specify info key
    #     if info_key is None:
    #         info_key = state.info_key
    #     else:
    #         info_key = f"{state.info_key}/info_key"
    #
    #     ## update info key
    #     old_info_keys = list(info.key())
    #     for k in old_info_keys:
    #         new_k = f"{info_key}/{k}"
    #         info[new_k] = info.pop(k)
    #
    #     return grad, info, policy_info["states"], policy_info["states_next"]

    # losses

    def target_state_loss(self, *args, **kwargs):
        return self._state_loss(
            *args,
            **kwargs,
            state_loss_fn=self.fake_state_loss_fn,
        )

    def target_policy_loss(self, *args, **kwargs):
        return self._policy_loss(
            *args,
            **kwargs,
            policy_loss_fn=self.fake_policy_loss_fn,
        )

    def source_state_loss(self, *args, **kwargs):
        return self._state_loss(
            *args,
            **kwargs,
            state_loss_fn=self.real_state_loss_fn,
        )

    def source_policy_loss(self, *args, **kwargs):
        return self._policy_loss(
            *args,
            **kwargs,
            policy_loss_fn=self.real_policy_loss_fn,
        )

    # # grad functions
    #
    # def target_grad_fn(self, *args, **kwargs):
    #     return self._grad_fn(
    #         *args,
    #         **kwargs,
    #         state_loss=self.target_state_loss,
    #         policy_loss=self.target_policy_loss,
    #         grads_processor=self.grads_processor,
    #     )
    #
    # def source_grad_fn(self, *args, **kwargs):
    #     return self._grad_fn(
    #         *args,
    #         **kwargs,
    #         state_loss=self.source_state_loss,
    #         policy_loss=self.source_policy_loss,
    #         grads_processor=self.grads_processor,
    #     )

class InDomainEncoderLoss(DomainEncoderLossMixin):
    def __call__(
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
        ts_loss, _ = self.target_state_loss(
            params=params,
            state=state,
            discriminator=state_discriminator,
            states=target_random_batch["observations"],
        )
        tp_loss, tp_info = self.target_policy_loss(
            params=params,
            state=state,
            discriminator=policy_discriminator,
            states=target_random_batch["observations"],
            states_next=target_random_batch["observations_next"],
        )
        ss_loss, _ = self.source_state_loss(
            params=params,
            state=state,
            discriminator=state_discriminator,
            states=source_expert_batch["observations"],
        )
        sp_loss, sp_info = self.source_policy_loss(
            params=params,
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

        return loss, {
            f"{state.info_key}/loss": loss,
            "target_random_batch": target_random_batch,
            "source_expert_batch": source_expert_batch
        }

# class InDomainEncoderGradFunc(DomainEncoderLossMixin):
#     def __call__(
#         self,
#         params: Params,
#         state: TrainState,
#         state_discriminator: Discriminator,
#         policy_discriminator: Discriminator,
#         target_random_batch: DataType,
#         source_expert_batch: DataType,
#         state_loss_scale: float,
#         *args,
#         **kwargs,
#     ):
#         (
#             target_grad,
#             target_info,
#             target_random_batch["observations"],
#             target_random_batch["observations_next"],
#         ) = self.target_grad_fn(
#             params=params,
#             state=state,
#             state_discriminator=state_discriminator,
#             policy_discriminator=policy_discriminator,
#             states=target_random_batch["observations"],
#             states_next=target_random_batch["observations_next"],
#             state_loss_scale=state_loss_scale,
#         )
#         (
#             source_grad,
#             source_info,
#             source_expert_batch["observations"],
#             source_expert_batch["observations_next"],
#         ) = self.source_grad_fn(
#             params=params,
#             state=state,
#             state_discriminator=state_discriminator,
#             policy_discriminator=policy_discriminator,
#             states=source_expert_batch["observations"],
#             states_next=source_expert_batch["observations_next"],
#             state_loss_scale=state_loss_scale,
#         )
#
#         grad = target_grad + source_grad
#         info = {
#             **target_info,
#             **source_info,
#             "target_random_batch": target_random_batch,
#             "source_expert_batch": source_expert_batch,
#         }
#         return grad, info
#
# class CrossDomainTargetEncoderGradFunc(DomainEncoderLossMixin):
#     def __call__(
#         self,
#         params: Params,
#         state: TrainState,
#         target_random_batch: DataType,
#         policy_discriminator: Discriminator,
#         state_discriminator: Discriminator,
#         state_loss_scale: float,
#         *args,
#         **kwargs,
#     ):
#         (
#             grad,
#             info,
#             target_random_batch["observations"],
#             target_random_batch["observations_next"],
#         ) = self.target_grad_fn(
#             params=params,
#             state=state,
#             state_discriminator=state_discriminator,
#             policy_discriminator=policy_discriminator,
#             states=target_random_batch["observations"],
#             states_next=target_random_batch["observations_next"],
#             state_loss_scale=state_loss_scale,
#         )
#         info["target_random_batch"] = target_random_batch
#         return grad, info
#
# class CrossDomainSourceEncoderGradFunc(DomainEncoderLossMixin):
#     def __call__(
#         self,
#         params: Params,
#         state: TrainState,
#         source_expert_batch: DataType,
#         policy_discriminator: Discriminator,
#         state_discriminator: Discriminator,
#         state_loss_scale: float,
#         *args,
#         **kwargs,
#     ):
#         (
#             grad,
#             info,
#             source_expert_batch["observations"],
#             source_expert_batch["observations_next"],
#         ) = self.target_grad_fn(
#             params=params,
#             state=state,
#             state_discriminator=state_discriminator,
#             policy_discriminator=policy_discriminator,
#             states=source_expert_batch["observations"],
#             states_next=source_expert_batch["observations_next"],
#             state_loss_scale=state_loss_scale,
#         )
#         info["source_expert_batch"] = source_expert_batch
#         return grad, info
