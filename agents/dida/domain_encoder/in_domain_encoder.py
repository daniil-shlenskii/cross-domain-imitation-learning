from .base_domain_encoder import BaseDomainEncoder


class InDomainEncoder(BaseDomainEncoder):
    def _update_encoder(self, learner_batch: DataType, expert_batch: DataType):
        new_learner_encoder, info, stats_info = self.learner_encoder.update(
            batch=learner_batch,
            expert_batch=expert_batch,
            policy_discriminator=self.policy_discriminator,
            state_discriminator=self.state_discriminator,
            state_loss_scale=self.state_loss_scale,
        )
        new_encoder = self.replace(
            learner_encoder=new_learner_encoder
        )
        return new_encoder, info, stats_info
