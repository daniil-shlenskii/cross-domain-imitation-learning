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
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        target_random_batch = deepcopy(target_random_batch)
        source_random_batch = deepcopy(source_random_batch)
        source_expert_batch = deepcopy(source_expert_batch)

        # state losses
        ts_loss, _ = self.target_state_loss(
            params=params,
            state=state,
            discriminator=state_discriminator,
            states=target_random_batch["observations"],
            states_next=target_random_batch["observations_next"],
        )
        ss_loss, _ = self.source_state_loss(
            params=params,
            state=state,
            discriminator=state_discriminator,
            states=source_expert_batch["observations"],
            states_next=source_expert_batch["observations_next"],
        )

        # policy losses
        trp_loss, trp_info = self.target_random_policy_loss(
            params=params,
            state=state,
            discriminator=policy_discriminator,
            states=target_random_batch["observations"],
            states_next=target_random_batch["observations_next"],
        )
        srp_loss, srp_info = self.source_random_policy_loss(
            params=params,
            state=state,
            discriminator=policy_discriminator,
            states=source_random_batch["observations"],
            states_next=source_random_batch["observations_next"],
        )
        sep_loss, sep_info = self.source_expert_policy_loss(
            params=params,
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

        return loss, {
            f"{state.info_key}/loss": loss,
            "target_random_batch": target_random_batch,
            "source_random_batch": source_random_batch,
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
    ):
        target_random_batch = deepcopy(target_random_batch)

        # state loss
        ts_loss, _ = self.target_state_loss(
            params=params,
            state=state,
            discriminator=state_discriminator,
            states=target_random_batch["observations"],
            states_next=target_random_batch["observations", "observations_next"],
        )

        # policy loss
        trp_loss, trp_info = self.target_random_policy_loss(
            params=params,
            state=state,
            discriminator=policy_discriminator,
            states=target_random_batch["observations"],
            states_next=target_random_batch["observations_next"],
        )

        # final loss
        loss = trp_loss + ts_loss * self.state_loss_scale

        # update batches
        target_random_batch["observations"] = trp_info["states"]
        target_random_batch["observations_next"] = trp_info["states_next"]

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
        source_random_batch: DataType,
        source_expert_batch: DataType,
        state_loss_scale: float,
    ):
        source_random_batch = deepcopy(source_random_batch)
        source_expert_batch = deepcopy(source_expert_batch)

        # state loss
        ss_loss, _ = self.source_state_loss(
            params=params,
            state=state,
            discriminator=state_discriminator,
            states=source_expert_batch["observations"],
            states_next=source_expert_batch["observations_next"],
        )

        # policy losses
        srp_loss, srp_info = self.source_expert_policy_loss(
            params=params,
            state=state,
            discriminator=policy_discriminator,
            states=source_random_batch["observations"],
            states_next=source_random_batch["observations_next"],
        )
        sep_loss, sep_info = self.source_expert_policy_loss(
            params=params,
            state=state,
            discriminator=policy_discriminator,
            states=source_expert_batch["observations"],
            states_next=source_expert_batch["observations_next"],
        )

        # final loss
        state_loss = ss_loss
        policy_loss = srp_loss + sep_loss
        loss = policy_loss + state_loss * self.state_loss_scale

        # update batches
        source_random_batch["observations"] = srp_info["states"]
        source_random_batch["observations_next"] = srp_info["states_next"]
        source_expert_batch["observations"] = sep_info["states"]
        source_expert_batch["observations_next"] = sep_info["states_next"]

        return loss, {
            f"{state.info_key}/source/loss": loss,
            "source_random_batch": source_random_batch,
            "source_expert_batch": source_expert_batch,
        }
