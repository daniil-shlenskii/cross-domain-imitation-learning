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


@jax.jit
def apply_model_jit(model: Callable, *args, **kwargs):
    return model(*args, **kwargs)

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
