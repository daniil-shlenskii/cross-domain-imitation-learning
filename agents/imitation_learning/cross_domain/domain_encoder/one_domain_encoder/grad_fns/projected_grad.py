import jax

from utils import project_a_to_b

from .base_grad_fn import BaseOneDomainEncoderGradFn


class OneDomainEncoderProjectedGradPolicyToStateGradFn(BaseOneDomainEncoderGradFn):
    def process_target_grads(self, *, state_grad, random_policy_grad):
        policy_to_state_projection = jax.tree.map(project_a_to_b, random_policy_grad, state_grad)
        random_policy_grad = jax.tree.map(lambda x, y: x - y, random_policy_grad, policy_to_state_projection)
        return state_grad, random_policy_grad

    def process_source_grads(self, *, state_grad, random_policy_grad, expert_policy_grad):
        expert_policy_to_state_projection = jax.tree.map(project_a_to_b, expert_policy_grad, state_grad)
        expert_policy_grad = jax.tree.map(lambda x, y: x - y, expert_policy_grad, expert_policy_to_state_projection)
        random_policy_to_state_projection = jax.tree.map(project_a_to_b, random_policy_grad, state_grad)
        random_policy_grad = jax.tree.map(lambda x, y: x - y, random_policy_grad, random_policy_to_state_projection)
        return state_grad, random_policy_grad, expert_policy_grad

class OneDomainEncoderProjectedGradStateToPolicyGradFn(BaseOneDomainEncoderGradFn):
    def process_target_grads(self, *, state_grad, random_policy_grad):
        state_to_policy_projection = jax.tree.map(project_a_to_b, state_grad, random_policy_grad)
        state_grad = jax.tree.map(lambda x, y: x - y, state_grad, state_to_policy_projection)
        return state_grad, random_policy_grad

    def process_source_grads(self, *, state_grad, random_policy_grad, expert_policy_grad):
        state_to_policy_projection = jax.tree.map(project_a_to_b, state_grad, expert_policy_grad)
        state_grad = jax.tree.map(lambda x, y: x - y, state_grad, state_to_policy_projection)
        return state_grad, random_policy_grad, expert_policy_grad
