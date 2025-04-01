import functools
from typing import Callable, Dict

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from hydra.utils import instantiate
from matplotlib.backends.backend_agg import FigureCanvasAgg
from omegaconf.dictconfig import DictConfig

from .custom_types import DataType, Params


@jax.jit
def encode_batch(encoder, batch: DataType):
    states, states_next = batch["observations"], batch["observations_next"]
    batch_size = states.shape[0]
    observations = jnp.concatenate([states, states_next])
    encoded_observations = encoder(observations)
    batch["observations"], batch["observations_next"] =\
        encoded_observations.at[:batch_size].get(), encoded_observations.at[batch_size:].get()
    return batch

@jax.jit
def encode_states_given_params(params: Params, state, states: jnp.ndarray, states_next: jnp.ndarray):
    batch_size = states.shape[0]
    observations = jnp.concatenate([states, states_next])
    encoded_observations = state.apply_fn({"params": params}, observations)
    return encoded_observations.at[:batch_size].get(), encoded_observations.at[batch_size:].get()

@jax.jit
def apply_model_jit(model: Callable, *args, **kwargs):
    return model(*args, **kwargs)

def flatten_params_fn(params_dict: dict):
    return jnp.concatenate([
        jnp.ravel(x) for x in
        jax.tree.flatten(params_dict, is_leaf=lambda x: isinstance(x, jnp.ndarray))[0]
    ])

def instantiate_optimizer(config: DictConfig):
    transforms = [
        instantiate(transform_config)
        for transform_config in config.transforms
    ]
    return optax.chain(*transforms)

def convert_figure_to_array(figure: plt.Figure) -> np.ndarray:
    agg = figure.canvas.switch_backends(FigureCanvasAgg)
    agg.draw()
    return np.asarray(agg.buffer_rgba())

def get_method_partially(path: str, params: Dict):
    method = hydra.utils.get_method(path)
    method = functools.partial(method, **params)
    return method
