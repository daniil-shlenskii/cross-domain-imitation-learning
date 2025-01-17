from copy import deepcopy

from agents.imitation_learning.cross_domain.domain_encoder.losses.mixin import \
    DomainEncoderLossMixin
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import DataType, Params


class OneDomainEncoderProjectedGradPolicyToStateGradFn(DomainEncoderLossMixin):
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

        # target
        target_losses, target_batches, target_grads = self.target_grad(...)

        # source 
        source_losses, source_batches, source_grads = self.source_grad(...)

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
