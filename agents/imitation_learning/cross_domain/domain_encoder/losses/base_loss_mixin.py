from agents.imitation_learning.cross_domain.domain_encoder.utils import \
    encode_states_given_params
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import DataType, Params

from .mixin import DomainEncoderLossMixin


class BaseDomainEncoderLossMixin(DomainEncoderLossMixin):
    def target_loss(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        target_random_batch: DataType,
    ):
        states, states_next = encode_states_given_params(
            params, state, target_random_batch["observations"], target_random_batch["observations_next"]
        )

        ts_loss = self.state_real_loss(
            states=states,
            discriminator=state_discriminator,
        )
        trp_loss = self.policy_fake_loss(
            states=states,
            states_next=states_next,
            discriminator=policy_discriminator,
        )

        # final loss
        loss = (
            trp_loss * self.target_policy_loss_scale * self.update_target_encoder_with_policy_discrminator +
            ts_loss * self.target_state_loss_scale
        )

        # update batches
        target_random_batch["observations"] = states
        target_random_batch["observations_next"] = states_next

        return loss, {"target_random_batch": target_random_batch}

    def source_loss(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        sr_states, sr_states_next = encode_states_given_params(
            params, state, source_random_batch["observations"], source_random_batch["observations_next"]
        )
        se_states, se_states_next = encode_states_given_params(
            params, state, source_expert_batch["observations"], source_expert_batch["observations_next"]
        )

        ss_loss = self.state_fake_loss(
            states=sr_states,
            discriminator=state_discriminator,
        )
        srp_loss = self.policy_fake_loss(
            states=sr_states,
            states_next=sr_states_next,
            discriminator=policy_discriminator,
        )
        sep_loss = self.policy_real_loss(
            states=sr_states,
            states_next=sr_states_next,
            discriminator=policy_discriminator,
        )

        # final loss
        state_loss = ss_loss
        policy_loss = srp_loss + sep_loss
        loss = (
            policy_loss *  self.source_policy_loss_scale +
            state_loss * self.source_state_loss_scale
        )

        # update batches
        source_random_batch["observations"] = sr_states
        source_random_batch["observations_next"] = sr_states_next
        source_expert_batch["observations"] = se_states
        source_expert_batch["observations_next"] = se_states_next

        return loss, {
            "source_random_batch": source_random_batch,
            "source_expert_batch": source_expert_batch,
        }
