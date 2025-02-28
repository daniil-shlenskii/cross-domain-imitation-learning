
import jax
import jax.numpy as jnp
from flax import struct

from .enot import ENOT


class ENOTGW(ENOT):
    source_mean: jnp.ndarray
    target_mean: jnp.ndarray
    proj_matrix: jnp.ndarray
    ema_decay: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        source_dim: int,
        target_dim: int,
        ema_decay: float=0.99,
        **kwargs,
    ):
        proj_matrix = jnp.zeros((target_dim, source_dim))
        kwargs["cost_fn_config"]["source_dim"] = source_dim
        kwargs["cost_fn_config"]["target_dim"] = target_dim

        return super().create(
            source_dim=source_dim,
            target_dim=target_dim,
            source_mean=jnp.zeros(source_dim),
            target_mean=jnp.zeros(target_dim),
            proj_matrix=proj_matrix,
            ema_decay=ema_decay,
            **kwargs
        )

    @jax.jit
    def __call__(self, source: jnp.ndarray):
        return self.transport(source - self.source_mean) + self.target_mean

    @jax.jit
    def update(self, target: jnp.ndarray, source: jnp.ndarray):
        # update means and cetralize batches
        target, new_target_mean = self.preprocess_batch(target, self.target_mean)
        source, new_source_mean = self.preprocess_batch(source, self.source_mean)

        # update cost fn
        new_proj_matrix = self.update_proj_matrix(target, source)
        new_cost_fn = self.cost_fn.replace(proj_matrix=new_proj_matrix)
        self = self.replace(proj_matrix=new_proj_matrix, cost_fn=new_cost_fn)

        enot, info, stats_info = super().update(target, source)

        enot = enot.replace(target_mean=new_target_mean, source_mean=new_source_mean)

        return enot, info, stats_info

    def preprocess_batch(self, batch: jnp.ndarray, mean: jnp.ndarray):
        new_mean = mean * self.ema_decay + batch.mean(0) * (1 - self.ema_decay)
        batch = batch - new_mean
        return batch, new_mean

    def update_proj_matrix(self, target: jnp.ndarray, source: jnp.ndarray):
        proj_matrix_opt = 4 * jax.vmap(lambda x, y: x[:, None] * y[None, :])(target, source).mean(0)
        return self.proj_matrix * self.ema_decay + proj_matrix_opt * (1 - self.ema_decay)
