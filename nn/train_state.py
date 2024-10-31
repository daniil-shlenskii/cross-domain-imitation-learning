from collections.abc import Callable

import pickle

import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState as FlaxTrainState


class TrainState(FlaxTrainState):
    loss_fn: Callable = struct.field(pytree_node=False)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({"params": self.params}, *args, **kwargs)

    def update(self, model_name: str, **loss_kwargs):
        grads, info = jax.grad(self.loss_fn, has_aux=True)(
            self.params, state=self, **loss_kwargs
        )

        stats_info = {}
        stats_info[f"{model_name}/max_grad_norm"] = _compute_norms(grads)
        stats_info[f"{model_name}/max_weight_norm"] = _compute_norms(self.params)

        return self.apply_gradients(grads=grads), info, stats_info

    def save(self, path: str) -> None:
        data = {
            "step": self.step,
            "params": self.params,
            "opt_state": self.opt_state,
        }
        with open(path, "wb") as file:
            pickle.dump(data, file)
    
    def load(self, path: str) -> "TrainState":
        with open(path, "rb") as file:
            data = pickle.load(file)
        return self.replace(**data)
    
def _compute_norms(pytree):
    norms = jax.tree.map(jnp.linalg.norm, pytree, is_leaf=lambda x: isinstance(x, jnp.ndarray))
    flatten_norms, _ = jax.tree.flatten(norms)
    return jnp.max(jnp.asarray(flatten_norms))
