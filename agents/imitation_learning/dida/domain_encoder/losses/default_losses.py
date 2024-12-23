from copy import deepcopy

from gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.types import DataType, Params

from .base import DomainEncoderLossMixin


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

class CrossDomainTargetEncoderLoss(DomainEncoderLossMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        target_random_batch: DataType,
        state_loss_scale: float,
    ):
        target_random_batch = deepcopy(target_random_batch)

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

        # final loss
        loss = tp_loss + ts_loss * state_loss_scale

        # update batches
        target_random_batch["observations"] = tp_info["states"]
        target_random_batch["observations_next"] = tp_info["states_next"]

        return loss, {
            f"{state.info_key}/target/loss": loss,
            "target_random_batch": target_random_batch,
        }

class CrossDomainSourceEncoderLoss(DomainEncoderLossMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        source_expert_batch: DataType,
        state_loss_scale: float,
    ):
        source_expert_batch = deepcopy(source_expert_batch)

        # compute state and policy losses
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
        loss = sp_loss + ss_loss * state_loss_scale

        # update batches
        source_expert_batch["observations"] = sp_info["states"]
        source_expert_batch["observations_next"] = sp_info["states_next"]

        return loss, {
            f"{state.info_key}/source/loss": loss,
            "source_expert_batch": source_expert_batch
        }
