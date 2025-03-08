from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax.struct import PyTreeNode
from typing_extensions import override


class BatchPreprocessor(PyTreeNode, ABC):

    @classmethod
    def create(cls, **kwargs):
        return cls()

    @override
    def update(self, batch: jnp.ndarray):
        return self, batch, {}

    @abstractmethod
    def encode(self, batch: jnp.ndarray):
        pass

    @abstractmethod
    def decode(self, batch: jnp.ndarray):
        pass

class IdentityPreprocessor(BatchPreprocessor):
    def encode(self, batch: jnp.ndarray):
        return batch

    def decode(self, batch: jnp.ndarray):
        return batch

class CentralizePreprocessor(BatchPreprocessor):
    mean: jnp.ndarray
    ema_decay: float

    @classmethod
    def create(cls, dim: int, ema_decay: float=0.99, **kwargs):
        return cls(mean=jnp.zeros(dim), ema_decay=ema_decay, **kwargs) 

    def update(self, batch: jnp.ndarray):
        new_mean = self.mean * self.ema_decay + batch.mean(0) * (1 - self.ema_decay)
        batch = batch - new_mean
        return self.replace(mean=new_mean), batch, {}

    def encode(self, batch: jnp.ndarray):
        return batch - self.mean

    def decode(self, batch: jnp.ndarray):
        return batch + self.mean

class CentralizeNormalizePreprocessor(CentralizePreprocessor):
    max_norm: jnp.ndarray

    @classmethod
    def create(cls, max_norm: float=1., **kwargs):
        return super().create(max_norm=max_norm, **kwargs) 

    def update(self, batch: jnp.ndarray):
        self, batch, _ = super().update(batch)
        batch_max_norm = jnp.max(jnp.linalg.norm(batch, axis=-1))
        new_max_norm = jnp.maximum(self.max_norm, batch_max_norm)
        batch /= new_max_norm
        return self.replace(max_norm=new_max_norm), batch, {}

    def encode(self, batch: jnp.ndarray):
        batch = super().encode(batch)
        return batch / self.max_norm

    def decode(self, batch: jnp.ndarray):
        batch *= self.max_norm
        return super().decode(batch)
