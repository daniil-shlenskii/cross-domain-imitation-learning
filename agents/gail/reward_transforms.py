import abc
from typing import Dict, List, Tuple

from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from jax import numpy as jnp

from utils.utils import SaveLoadFrozenDataclassMixin


class RewardTransform(PyTreeNode, SaveLoadFrozenDataclassMixin):
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, **kwargs):
        if "_save_attrs" not in kwargs:
            kwargs["_save_attrs"] = tuple()
        return cls(**kwargs)

    @abc.abstractmethod
    def transform(self, rewards: jnp.ndarray) -> jnp.ndarray:
        ...

    @abc.abstractmethod
    def update(self, rewards: jnp.ndarray) -> "RewardTransform":
        ...

class RewardTransformList(PyTreeNode):
    transforms: List[RewardTransform]

    @classmethod
    def create(cls, tranforms_config: List[Dict]) -> RewardTransform:
        transforms = [
            instantiate(tranform_config)
            for tranform_config in tranforms_config
        ]
        return cls(
            transforms=transforms,
            _save_attrs=("transforms",)
        )

    def transform(self, rewards: jnp.ndarray) -> jnp.ndarray:
        for transform in self.transforms:
            rewards = transform(rewards)
        return rewards
        
    def update(self, rewards: jnp.ndarray) -> RewardTransform:
        new_transforms = []
        for transform in self.transform:
            new_transform = transform.update(rewards)
            rewards = new_transform.transform(rewards)
            new_transforms.append(new_transform)
        return self.replace(transforms=new_transforms)

class IdentityRewardTransform(RewardTransform):
    def transform(self, rewards: jnp.ndarray) -> jnp.ndarray:
        return rewards

    def update(self, rewards: jnp.ndarray) -> "RewardTransform":
        return self

class RewardStandartization(RewardTransform):
    mean: jnp.ndarray
    std: jnp.ndarray
    ema: float
    eps: float

    @classmethod
    def create(
        cls, mean=0., std=1., ema=0.99, eps=1e-8,
    ) -> RewardTransform:
        return cls(
            mean=mean,
            std=std,
            ema=ema,
            eps=eps,
            _save_attrs=("mean", "std", "ema", "eps")
        )

    def transform(self, rewards: jnp.ndarray) -> jnp.ndarray:
        return (rewards - self.mean) / (self.std + self.eps)
    
    def update(self, rewards: jnp.ndarray) -> RewardTransform:
        new_mean = self.mean * self.ema + rewards.mean() * (1 - self.ema)
        new_std = self.std * self.ema + rewards.std() * (1 - self.ema)
        return self.replace(mean=new_mean, std=new_std)