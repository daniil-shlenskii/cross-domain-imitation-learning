import abc

import numpy as np


class DomainLossScale(abc.ABC):
    @abc.abstractmethod
    def __call__(self, dida_agent: "DIDAAgent"):
        pass

class ConstantDomainLossScale(DomainLossScale):
    def __init__(self, domain_loss_scale: float=1.0):
        self.domain_loss_scale = domain_loss_scale

    def __call__(self, dida_agent: "DIDAAgent") -> float:
        return self.domain_loss_scale

class LinearDomainLossScale(DomainLossScale):
    def __init__(
        self,
        start_scale: float = 0.,
        end_scale: float = 0.5,
        max_n_iters: int = 50_000,
    ):
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.max_n_iters = max_n_iters

    def __call__(self, dida_agent: "DIDAAgent") -> float:
        curr_iter = dida_agent.learner_encoder.state.step
        if curr_iter >= self.max_n_iters:
            return self.end_scale

        domain_loss_scale = self.start_scale + (self.end_scale - self.start_scale) * curr_iter / self.max_n_iters
        return domain_loss_scale

class ExponentialDomainLossScale(DomainLossScale):
    def __init__(
        self,
        start_scale: float = 0.,
        end_scale: float = 0.5,
        max_n_iters: int = 50_000,
        concavity: float = 10.,
    ):
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.max_n_iters = max_n_iters
        self.concavity = concavity

    def __call__(self, dida_agent: "DIDAAgent") -> float:
        curr_iter = dida_agent.learner_encoder.state.step
        if curr_iter >= self.max_n_iters:
            return self.end_scale

        q = curr_iter / self. max_n_iters
        domain_loss_scale = (
            self.start_scale +
            self.end_scale * (
                2 / (1 + np.exp(-self.concavity * q)) - 1
            )
        )
        return domain_loss_scale
