from collections.abc import Callable
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import core, struct
from flax.training.train_state import TrainState as FlaxTrainState

from utils.utils import SaveLoadFrozenDataclassMixin


class TrainState(FlaxTrainState, SaveLoadFrozenDataclassMixin):
    united_grads: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    loss_fn: Callable = struct.field(pytree_node=False)
    info_key: str = struct.field(pytree_node=False)
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        return super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            united_grads=jax.tree.map(lambda x: 0, params),
            _save_attrs=("step", "params", "opt_state"),
            **kwargs,
        )

    def apply_gradients(self, *, grads, **kwargs):
        return super().apply_gradients(
            grads=grads,
            united_grads=jax.tree.map(lambda x: 0, grads),
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn({"params": self.params}, *args, **kwargs)

    def update(self, hold_grad=False, **loss_kwargs):
        grads, info = jax.grad(self.loss_fn, has_aux=True)(
            self.params, state=self, **loss_kwargs
        )

        stats_info = {}
        stats_info[f"{self.info_key}/max_grad_norm"] = _compute_norms(grads)
        stats_info[f"{self.info_key}/max_weight_norm"] = _compute_norms(self.params)

        if hold_grad:
            new_united_grads = self.united_grads + grads
            return self.replace(united_grads=new_united_grads), info, stats_info    
        return self.apply_gradients(grads=grads), info, stats_info
    
def _compute_norms(pytree):
    norms = jax.tree.map(jnp.linalg.norm, pytree, is_leaf=lambda x: isinstance(x, jnp.ndarray))
    flatten_norms, _ = jax.tree.flatten(norms)
    return jnp.max(jnp.asarray(flatten_norms))
