import abc

import jax.numpy as jnp
from flax.struct import PyTreeNode


class DomainLossScale(PyTreeNode):
    @abc.abstractmethod
    def __call__(self, dida_agent: "DIDAAgent"):
        pass

class ConstantDomainLossScale(DomainLossScale):
    domain_loss_scale: float

    def __call__(self, dida_agent: "DIDAAgent") -> float:
        return self.domain_loss_scale

class LinearDomainLossScale(DomainLossScale):
    start_scale: float
    end_scale: float
    max_n_iters: int

    def __call__(self, dida_agent: "DIDAAgent") -> float:
        curr_iter = dida_agent.learner_encoder.state.step
        domain_loss_scale = self.start_scale + (self.end_scale - self.start_scale) * curr_iter / self.max_n_iters
        return jnp.minimum(domain_loss_scale, self.end_scale)

class ExponentialDomainLossScale(DomainLossScale):
    start_scale: float
    end_scale: float
    max_n_iters: int
    concavity: float

    def __call__(self, dida_agent: "DIDAAgent") -> float:
        curr_iter = dida_agent.learner_encoder.state.step
        q = curr_iter / self. max_n_iters
        domain_loss_scale = (
            self.start_scale +
            self.end_scale * (
                2 / (1 + jnp.exp(-self.concavity * q)) - 1
            )
        )
        return jnp.minimum(domain_loss_scale, self.end_scale)
