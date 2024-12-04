from typing import Tuple

from flax import struct
from flax.struct import PyTreeNode
from jax import numpy as jnp
from typing_extensions import override

from utils import SaveLoadFrozenDataclassMixin


class BaseRewardTransform(PyTreeNode, SaveLoadFrozenDataclassMixin):
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, **kwargs):
        if "_save_attrs" not in kwargs:
            kwargs["_save_attrs"] = tuple()
        return cls(**kwargs)

    @override
    def transform(self, rewards: jnp.ndarray) -> jnp.ndarray:
        return rewards

    @override
    def update(self, rewards: jnp.ndarray) -> "BaseRewardTransform":
        return self, {}

class RewardStandartization(BaseRewardTransform):
    mean: jnp.ndarray
    std: jnp.ndarray
    ema: float
    eps: float

    @classmethod
    def create(
        cls, *, mean=0., std=1., ema=0.99, eps=1e-8,
    ) -> BaseRewardTransform:
        return cls(
            mean=mean,
            std=std,
            ema=ema,
            eps=eps,
            _save_attrs=("mean", "std")
        )

    def transform(self, rewards: jnp.ndarray) -> jnp.ndarray:
        return (rewards - self.mean) / (self.std + self.eps)

    def update(self, rewards: jnp.ndarray) -> BaseRewardTransform:
        new_mean = self.mean * self.ema + rewards.mean() * (1 - self.ema)
        new_std = self.std * self.ema + rewards.std() * (1 - self.ema)
        info = {
            "reward_standartization/mean": new_mean,
            "reward_standartization/std": new_std,
        }
        return self.replace(mean=new_mean, std=new_std), info
