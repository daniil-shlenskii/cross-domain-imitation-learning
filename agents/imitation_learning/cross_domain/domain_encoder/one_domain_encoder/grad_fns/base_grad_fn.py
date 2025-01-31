import jax

from agents.imitation_learning.cross_domain.domain_encoder.grad_fns import \
    BaseDomainEncoderGradFnMixin
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import DataType, Params


class BaseOneDomainEncoderGradFn(BaseDomainEncoderGradFnMixin):
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
        #
        target_grad, target_loss, target_batches = self.target_grad(
            params=params,
            state=state,
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            random_batch=target_random_batch,
        )
        source_grad, source_loss, source_batches = self.source_grad(
            params=params,
            state=state,
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            random_batch=source_random_batch,
            expert_batch=source_expert_batch,
        )

        #
        grad = jax.tree.map(lambda x, y: x + y, target_grad, source_grad)
        loss = target_loss + source_loss

        return grad, {
            f"{state.info_key}/loss": loss,
            **target_batches,
            **source_batches,
        }
