import jax.numpy as jnp
from flax.struct import PyTreeNode


class GWCost(PyTreeNode):
    proj_matrix: jnp.ndarray

    @classmethod
    def create(cls, source_dim: int, target_dim: int):
        proj_matrix = jnp.zeros((target_dim, source_dim))
        return cls(proj_matrix=proj_matrix)

    def __call__(self, x, y):
        add1 = (y * (self.proj_matrix @ x)).sum()
        add2 = jnp.linalg.norm(x)**2 * jnp.linalg.norm(y)**2
        return -(add1 + add2)

class GWCostStable(PyTreeNode):
    proj_matrix: jnp.ndarray
    c: int

    @classmethod
    def create(cls, source_dim: int, target_dim: int, c: int=None):
        proj_matrix = jnp.zeros((target_dim, source_dim))
        if c is None:
            c = source_dim**0.5
        return cls(proj_matrix=proj_matrix, c=c)

    def __call__(self, x, y):
        add1 = 0.5 * jnp.linalg.norm(self.proj_matrix @ x - y)**2
        add2 = (self.c - jnp.linalg.norm(x)**2) * jnp.linalg.norm(y)**2
        return add1 + add2
