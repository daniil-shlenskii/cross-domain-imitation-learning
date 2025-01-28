from agents.imitation_learning.cross_domain.domain_encoder.losses.mixin import \
    DomainEncoderLossMixin
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import DataType, Params


class UDILSourceDomainEncoderLoss(DomainEncoderLossMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        policy_discriminator: Discriminator,
        source_random_batch: DataType,
        source_expert_batch: DataType,
        **kwargs,
    ):
        srp_loss, srp_info = self.policy_real_loss(
            params=params,
            state=state,
            discriminator=policy_discriminator,
            states=source_random_batch["observations"],
            states_next=source_random_batch["observations_next"],
        )
        sep_loss, sep_info = self.policy_fake_loss(
            params=params,
            state=state,
            discriminator=policy_discriminator,
            states=source_expert_batch["observations"],
            states_next=source_expert_batch["observations_next"],
        )

        # final loss
        loss = srp_loss + sep_loss

        # update batches
        source_random_batch["observations"] = srp_info["states"]
        source_random_batch["observations_next"] = srp_info["states_next"]
        source_expert_batch["observations"] = sep_info["states"]
        source_expert_batch["observations_next"] = sep_info["states_next"]

        return loss, {
            "source_random_batch": source_random_batch,
            "source_expert_batch": source_expert_batch,
        }
