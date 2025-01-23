from agents.imitation_learning.cross_domain.domain_encoder.losses.base_loss_mixin import \
    BaseDomainEncoderLossMixin
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import DataType, Params


class BaseTargetDomainEncoderLoss(BaseDomainEncoderLossMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        target_random_batch: DataType,
    ):
        loss, batches = self.target_loss(
            params,
            state=state,
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            target_random_batch=target_random_batch
        )

        return loss, {
            f"{state.info_key}/target/loss": loss,
            **batches
        }

class BaseSourceDomainEncoderLoss(BaseDomainEncoderLossMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        loss, batches = self.source_loss(
            params,
            state=state,
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            source_random_batch=source_random_batch,
            source_expert_batch=source_expert_batch,
        )
        return loss, {
            f"{state.info_key}/source/loss": loss,
            **batches,
        }
