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
        update_range: float=0.05,
        factor: float=0.5,
        min_clip: float = 0.1,
    ):
        self.update_range = update_range
        self.factor = factor
        self.min_clip = min_clip
        self._updated_recently = False

    def update(self, dida_agent: "DIDAAgent") -> float:
        domain_loss_scale = dida_agent.domain_loss_scale
        update_condition = (
            0.5 - self.update_range <= dida_agent.p_acc_ema < 0.5 + self.update_range
        )
        if self._updated_recently:
            if not update_condition:
                self._updated_recently = False
        else:
            if update_condition:
                domain_loss_scale = domain_loss_scale * self.factor
                self._updated_recently = True
        return np.clip(domain_loss_scale, self.min_clip, None)
