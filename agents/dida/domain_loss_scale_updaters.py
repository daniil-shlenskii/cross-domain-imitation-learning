import abc

import numpy as np


class DomainLossScaleUpdater(abc.ABC):
    @abc.abstractmethod
    def update(self, dida_agent: "DIDAAgent") -> float:
        pass

class IdentityDomainLossScaleUpdater(abc.ABC):
    def update(self, dida_agent: "DIDAAgent") -> float:
        return dida_agent.domain_loss_scale

class RangeDomainLossScaleUpdater:
    def __init__(
        self,
        update_range: float = 0.03,
        unlock_range: float = 0.1,
        update_factor: float = 0.5,
        min_clip: float = 0.1,
    ):
        self.update_range = update_range
        self.unlock_range = unlock_range

        self.update_factor = update_factor

        self.min_clip = min_clip
        self._updated_recently = False

    def update(self, dida_agent: "DIDAAgent") -> float:
        domain_loss_scale = dida_agent.domain_loss_scale
        update_condition = (
            0.5 - self.update_range <= dida_agent.p_acc_ema < 0.5 + self.update_range
        )
        if self._updated_recently:
            unlock_condition = not (
                0.5 - self.unlock_range <= dida_agent.p_acc_ema < 0.5 + self.unlock_range
            )
            if unlock_condition:
                self._updated_recently = False
        else:
            if update_condition:
                domain_loss_scale = domain_loss_scale * self.update_factor
                self._updated_recently = True
        return np.clip(domain_loss_scale, self.min_clip, None)


class LinearLossScaleUpdater:
    def __init__(
        self,
        start_scale: float = 0.,
        end_scale: float = 0.5,
        max_n_iters: int = 500_000,
    ):
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.max_n_iters = max_n_iters

    def update(self, dida_agent: "DIDAAgent") -> float:
        curr_iter = dida_agent.learner_encoder.state.step
        domain_loss_scale = self.start_scale + (self.end_scale - self.start_scale) * curr_iter / self.max_n_iters
        return domain_loss_scale

class ExponentialLossScaleUpdater:
    def __init__(
        self,
        start_scale: float = 0.,
        end_scale: float = 0.5,
        concavity_param: float = 10., 
        max_n_iters: int = 1_000_000,
    ):
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.concavity_param = concavity_param
        self.max_n_iters = max_n_iters

    def update(self, dida_agent: "DIDAAgent") -> float:
        curr_iter = dida_agent.learner_encoder.state.step
        q = curr_iter / self. max_n_iters
        domain_loss_scale = (
            self.start_scale + 
            self.end_scale * (
                2 / (1 + np.exp(-self.concavity_param * q))
                -
                1
            )
        )
        if self.start_scale < self.end_scale:
            return min(domain_loss_scale, self.end_scale)
        return max(domain_loss_scale, self.end_scale)
