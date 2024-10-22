import jax
from typing import Callable
from flax.training.train_state import TrainState as _TrainState


class TrainState(_TrainState):
    def __init__(
        self, *, loss_fn: Callable, **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def update(
        self, **loss_kwargs,
    ):
        grads = jax.grad(self.loss_fn)(params=self.params, **loss_kwargs)
        return self.apply_gradients(grads=grads)
