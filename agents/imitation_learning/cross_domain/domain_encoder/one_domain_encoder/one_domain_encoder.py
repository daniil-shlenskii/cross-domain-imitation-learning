from agents.imitation_learning.cross_domain.domain_encoder.base_domain_encoder import \
    BaseDomainEncoder
from utils.custom_types import DataType


class OneDomainEncoder(BaseDomainEncoder):
    def _update_encoder(
        self,
        *,
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        new_target_encoder, info, stats_info = self.target_encoder.update(
            target_random_batch=target_random_batch,
            source_random_batch=source_random_batch,
            source_expert_batch=source_expert_batch,
            policy_discriminator=self.policy_discriminator,
            state_discriminator=self.state_discriminator,
        )
        new_encoder = self.replace(target_encoder=new_target_encoder)
        return new_encoder, info, stats_info
