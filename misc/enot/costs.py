import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode


class GWCost(PyTreeNode):
    proj_matrix: jnp.ndarray
    ema_decay: float = struct.field(pytree_node=False)

    @classmethod
    def create(cls, source_dim: int, target_dim: int, ema_decay: float=0.99, **kwargs):
        proj_matrix = jnp.eye(target_dim, source_dim)
        return cls(proj_matrix=proj_matrix, ema_decay=ema_decay)

    def __call__(self, x, y):
        add1 = (y * (self.proj_matrix @ x)).sum()
        add2 = jnp.linalg.norm(x)**2 * jnp.linalg.norm(y)**2
        return -(add1 + add2)

    def update(self, *, source: jnp.ndarray, target: jnp.ndarray):
        proj_matrix_opt = 4 * jax.vmap(lambda x, y: x[:, None] * y[None, :])(target, source).mean(0)
        new_proj_matrix = self.proj_matrix * self.ema_decay + proj_matrix_opt * (1 - self.ema_decay)
        return self.replace(proj_matrix=new_proj_matrix)

class GWCostStable(GWCost):
    c: jnp.ndarray 

    @classmethod
    def create(cls, source_dim: int, target_dim: int, c: int=None):
        if c is None:
            c = 1.
        return super().create(source_dim=source_dim, target_dim=target_dim, c=c)

    def __call__(self, x, y):
        add1 = 0.5 * jnp.linalg.norm(self.proj_matrix @ x - y)**2
        add2 = (self.c - jnp.linalg.norm(x)**2) * jnp.linalg.norm(y)**2
        return add1 + add2

class InnerGWCost(PyTreeNode):
    proj_matrix: jnp.ndarray
    source_dim: int
    ema_decay: float = struct.field(pytree_node=False)

    @classmethod
    def create(cls, source_dim: int, target_dim: int, ema_decay: float=0.99, **kwargs):
        proj_matrix = jnp.eye(target_dim, source_dim)
        return cls(source_dim=source_dim, proj_matrix=proj_matrix, ema_decay=ema_decay)

    def __call__(self, x, y):
        return -(y * (self.proj_matrix @ x)).sum()

    def update(self, *, source: jnp.ndarray, target: jnp.ndarray):
        proj_matrix_opt = jax.vmap(lambda x, y: x[:, None] * y[None, :])(target, source).mean(0)
        proj_matrix_opt = self.source_dim**0.5 * proj_matrix_opt / jnp.linalg.norm(proj_matrix_opt)
        new_proj_matrix = self.proj_matrix * self.ema_decay + proj_matrix_opt * (1 - self.ema_decay)
        new_proj_matrix = self.source_dim**0.5 * new_proj_matrix / jnp.linalg.norm(new_proj_matrix)
        return self.replace(proj_matrix=new_proj_matrix)

class InnerGWCostStable(InnerGWCost):
    def __call__(self, x, y):
        return jnp.linalg.norm(self.proj_matrix @ x - y)**2

class PairsCost(PyTreeNode):
    state_cost: InnerGWCostStable
    pairs_cost: InnerGWCostStable

    def __getattr__(self, name):
        return self.pairs_cost.__getattribute__(name)

    @classmethod
    def create(cls, source_dim: int, target_dim: int, ema_decay: float=0.99, **kwargs):
        state_cost = InnerGWCostStable.create(source_dim=source_dim//2, target_dim=target_dim//2, ema_decay=ema_decay)
        pairs_cost = InnerGWCostStable.create(source_dim=source_dim, target_dim=target_dim, ema_decay=ema_decay)
        return cls(state_cost=state_cost, pairs_cost=pairs_cost)

    def __call__(self, x_pairs, y_pairs):
        x = jnp.split(x_pairs, 2, axis=-1)[0]
        y = jnp.split(y_pairs, 2, axis=-1)[0]
        return self.state_cost(x, y) + self.pairs_cost(x_pairs, y_pairs)

    def update(self, *, source: jnp.ndarray, target: jnp.ndarray):
        source_pairs, target_pairs = source, target
        source = jnp.split(source_pairs, 2, axis=-1)[0] 
        target = jnp.split(target_pairs, 2, axis=-1)[0]
        new_state_cost = self.state_cost.update(source=source, target=target)
        new_pairs_cost = self.pairs_cost.update(source=source_pairs, target=target_pairs)
        return self.replace(state_cost=new_state_cost, pairs_cost=new_pairs_cost)
