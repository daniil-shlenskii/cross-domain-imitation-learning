from agents.imitation_learning.cross_domain.domain_encoder.grad_fns import \
    DomainEncoderProjectedGradFnMixin

from .base_grad_fn import BaseOneDomainEncoderGradFn


class OneDomainEncoderPolicyToStateProjectedGradFn(BaseOneDomainEncoderGradFn, DomainEncoderProjectedGradFnMixin):
    def process_target_grads(self, *, state_grad, random_policy_grad):
        random_policy_grad = self.remove_grad_projection_to_b_from_a(random_policy_grad, state_grad)
        return state_grad, random_policy_grad

    def process_source_grads(self, *, state_grad, random_policy_grad, expert_policy_grad):
        random_policy_grad = self.remove_grad_projection_to_b_from_a(random_policy_grad, state_grad)
        expert_policy_grad = self.remove_grad_projection_to_b_from_a(expert_policy_grad, state_grad)
        return state_grad, random_policy_grad, expert_policy_grad
