from collections.abc import Callable
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState as FlaxTrainState

from utils import SaveLoadFrozenDataclassMixin


class TrainState(FlaxTrainState, SaveLoadFrozenDataclassMixin):
    loss_fn: Optional[Callable] = struct.field(pytree_node=False)
    grad_fn: Optional[Callable] = struct.field(pytree_node=False)
    info_key: str = struct.field(pytree_node=False)
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, *, apply_fn, params, tx, loss_fn: Callable = None, grad_fn: Callable = None, **kwargs
    ):
        if grad_fn is None:
            assert loss_fn is not None
            def grad_fn(*, params, state, **loss_kwargs):
                grads, info = jax.grad(loss_fn, has_aux=True)(
                    params, state=state, **loss_kwargs
                )
                return grads, info

        return super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            loss_fn=loss_fn,
            grad_fn=grad_fn,
            _save_attrs=("step", "params", "opt_state"),
            **kwargs,
        )

    def __getattribute__(self, item: str):
        if self.loss_fn is None and item == "loss_fn":
            return self.grad_fn
        return super().__getattribute__(item)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({"params": self.params}, *args, **kwargs)

    def update(self, **loss_kwargs):
        grads, info = self.grad_fn(self.params, state, **loss_kwargs)

        stats_info = {}
        stats_info[f"{self.info_key}/max_grad_norm"] = _compute_norms(grads)
        stats_info[f"{self.info_key}/max_weight_norm"] = _compute_norms(self.params)

        new_state = self.apply_gradients(grads=grads)
        return new_state, info, stats_info

def _compute_norms(pytree):
    norms = jax.tree.map(jnp.linalg.norm, pytree, is_leaf=lambda x: isinstance(x, jnp.ndarray))
    flatten_norms, _ = jax.tree.flatten(norms)
    return jnp.max(jnp.asarray(flatten_norms))
