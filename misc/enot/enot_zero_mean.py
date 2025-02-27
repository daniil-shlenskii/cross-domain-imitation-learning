from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

from utils.custom_types import DataType

from .enot import ENOT


class ENOTZeroMean(ENOT):
    source_mean: jnp.ndarray
    target_mean: jnp.ndarray
    ema_decay: float = struct.field(pytree_node=False)
    batch_preprocessor: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        source_dim: int,
        target_dim: int,
        batch_is_dict: bool=False,
        ema_decay: float=0.99,
        **kwargs,
    ):
        if batch_is_dict:
            def batch_preprocessor(batch: DataType, mean: jnp.ndarray):
                new_mean = mean * ema_decay + batch["observations"].mean(0) * (1 - ema_decay)
                batch["observations"] = batch["observations"] - new_mean
                batch["observations_next"] = batch["observations_next"] - new_mean
                return batch, new_mean
        else:
            def batch_preprocessor(batch: jnp.ndarray, mean: jnp.ndarray):
                new_mean = mean * ema_decay + batch.mean(0) * (1 - ema_decay)
                batch = batch - new_mean
                return batch, new_mean

        return super().create(
            source_dim=source_dim,
            target_dim=target_dim,
            source_mean=jnp.zeros(source_dim),
            target_mean=jnp.zeros(target_dim),
            ema_decay=ema_decay,
            batch_preprocessor=batch_preprocessor,
            **kwargs
        )

    @jax.jit
    def __call__(self, source: jnp.ndarray):
        return self.transport(source - self.source_mean) + self.target_mean

    @jax.jit
    def update(self, target: jnp.ndarray, source: jnp.ndarray):
        target, new_target_mean = self.batch_preprocessor(target, self.target_mean)
        source, new_source_mean = self.batch_preprocessor(source, self.source_mean)

        enot, info, stats_info = super().update(target, source)

        enot = enot.replace(target_mean=new_target_mean, source_mean=new_source_mean)

        return enot, info, stats_info
