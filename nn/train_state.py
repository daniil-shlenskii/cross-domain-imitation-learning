import pickle
from collections.abc import Callable
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState as FlaxTrainState

from utils.utils import SaveLoadFrozenDataclassMixin


class TrainState(FlaxTrainState, SaveLoadFrozenDataclassMixin):
    loss_fn: Callable = struct.field(pytree_node=False)
    info_key: str = struct.field(pytree_node=False)
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, **kwargs):
        kwargs["_save_attrs"] = ("step", "params", "opt_state")
        return super().create(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({"params": self.params}, *args, **kwargs)

    def update(self, **loss_kwargs):
        grads, info = jax.grad(self.loss_fn, has_aux=True)(
            self.params, state=self, **loss_kwargs
        )

        stats_info = {}
        stats_info[f"{self.info_key}/max_grad_norm"] = _compute_norms(grads)
        stats_info[f"{self.info_key}/max_weight_norm"] = _compute_norms(self.params)

        return self.apply_gradients(grads=grads), info, stats_info
    
def _compute_norms(pytree):
    norms = jax.tree.map(jnp.linalg.norm, pytree, is_leaf=lambda x: isinstance(x, jnp.ndarray))
    flatten_norms, _ = jax.tree.flatten(norms)
    return jnp.max(jnp.asarray(flatten_norms))
