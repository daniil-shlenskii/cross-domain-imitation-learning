import jax
import jax.numpy as jnp

from utils.types import DataType

from .base_domain_encoder import BaseDomainEncoder


class InDomainEncoder(BaseDomainEncoder):
    @jax.jit
    def encode_source_state(self, state: jnp.ndarray):
        return self.learner_encoder(state)

    def _update_encoder(
        self,
        *,
        target_random_batch: DataType,
        target_expert_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
   ):
        new_learner_encoder, info, stats_info = self.learner_encoder.update(
            target_random_batch=target_random_batch,
            target_expert_batch=target_expert_batch,
            source_random_batch=source_random_batch,
            source_expert_batch=source_expert_batch,
            policy_discriminator=self.policy_discriminator,
            state_discriminator=self.state_discriminator,
            state_loss_scale=self.state_loss_scale,
        )
        new_encoder = self.replace(learner_encoder=new_learner_encoder)
        return new_encoder, info, stats_info
