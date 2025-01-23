from agents.imitation_learning.cross_domain.domain_encoder.losses.base_loss_mixin import \
    BaseDomainEncoderLossMixin
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import DataType, Params


class BaseOneDomainEncoderLoss(BaseDomainEncoderLossMixin):
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
        target_loss, target_batches = self.target_loss(
            params,
            state=state,
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            target_random_batch=target_random_batch,
        )

        source_loss, source_batches = self.source_loss(
            params,
            state=state,
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            source_random_batch=source_random_batch,
            source_expert_batch=source_expert_batch,
        )

        loss = (target_loss + source_loss) * 0.5

        return loss, {
            f"{state.info_key}/loss": loss,
            f"{state.info_key}/target/loss": target_loss,
            f"{state.info_key}/source/loss": source_loss,
            **target_batches,
            **source_batches,
        }
