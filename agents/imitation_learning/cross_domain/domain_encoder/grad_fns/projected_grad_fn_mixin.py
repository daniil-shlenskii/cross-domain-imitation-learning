import jax

from utils import project_a_to_b

from .base_grad_fn_mixin import BaseDomainEncoderGradFnMixin


class DomainEncoderProjectedGradFnMixin(BaseDomainEncoderGradFnMixin):
    def project_grad_a_to_b(self, grad_a, grad_b):
        return jax.tree.map(project_a_to_b, grad_a, grad_b)

    def remove_grad_projection_to_b_from_a(self, grad_a, grad_b):
        projection_a_to_b = self.project_grad_a_to_b(grad_a, grad_b)
        return jax.tree.map(lambda x, y: x - y, grad_a, projection_a_to_b)
