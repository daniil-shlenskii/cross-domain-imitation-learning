import jax

from agents.imitation_learning.cross_domain.domain_encoder.grad_fns import \
    BaseDomainEncoderGradFnMixin
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import DataType, Params


class BaseTargetEncoderGradFn(BaseDomainEncoderGradFnMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        target_random_batch: DataType,
    ):
        #
        grad, loss, batches = self.target_grad(
            params=params,
            state=state,
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            random_batch=target_random_batch,
        )

        return grad, {
            f"{state.info_key}/target/loss": loss,
            **batches,
        }

class BaseSourceEncoderGradFn(BaseDomainEncoderGradFnMixin):
    def __call__(
        self,
        params: Params,
        state: TrainState,
        state_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        grad, loss, batches = self.source_grad(
            params=params,
            state=state,
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            random_batch=source_random_batch,
            expert_batch=source_expert_batch,
        )

        return grad, {
            f"{state.info_key}/source/loss": loss,
            **batches,
        }
