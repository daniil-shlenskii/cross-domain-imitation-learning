import abc
from typing import List, Dict

from hydra.utils import instantiate

import jax
from jax import numpy as jnp
from flax.struct import PyTreeNode
from jax.numpy import ndarray


class RewardTransform(PyTreeNode):
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
        return cls(transforms=transforms)

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
    mean: jnp.ndarray = 0.
    std: jnp.ndarray = 1.
    ema: float = 0.99
    eps: float = 1e-8

    def transform(self, rewards: jnp.ndarray) -> jnp.ndarray:
        return (rewards - self.mean) / (self.std + self.eps)
    
    def update(self, rewards: jnp.ndarray) -> RewardTransform:
        new_mean = self.mean * self.ema + rewards.mean() * (1 - self.ema)
        new_std = self.std * self.ema + rewards.std() * (1 - self.ema)
        return self.replace(mean=new_mean, std=new_std)