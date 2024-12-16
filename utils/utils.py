import functools
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Callable, Dict

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from hydra.utils import instantiate
from matplotlib.backends.backend_agg import FigureCanvasAgg
from omegaconf.dictconfig import DictConfig

from utils.types import Buffer, BufferState, PRNGKey


@jax.jit
def apply_model_jit(model: Callable, input: jnp.ndarray):
    return model(input)

def instantiate_optimizer(config: DictConfig):
    transforms = [
        instantiate(transform_config)
        for transform_config in config.transforms
    ]
    return optax.chain(*transforms)

def save_json(data, path: str):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

def load_json(path: str):
    with open(path) as file:
        data = json.load(data, file)
    return data

def save_pickle(data: Any, path: str):
    with open(path, "wb") as file:
        pickle.dump(data, file)

def load_pickle(path: str) -> Any:
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data

def load_buffer(state: Buffer, path: str, size: int = None):
    stored_state = load_pickle(path)
    if state.experience.keys() == stored_state.experience.keys():
        # define states sized
        stored_state_size = get_buffer_state_size(stored_state)
        state_max_size = state.experience["observations"][0].shape[0]
        data_size = min(stored_state_size, state_max_size)
        if size is not None:
            data_size = min(data_size, size)

        # create epxerience for new state
        stored_state_exp = stored_state.experience
        state_exp = state.experience
        for k, v in stored_state_exp.items():
            state_exp[k] = \
                state_exp[k].at[0, :data_size].set(v[0, :data_size])

        # define current index for new state
        if data_size == state_max_size:
            current_index = 0
            is_full = True
        else:
            current_index = data_size
            is_full = False

        state = state.replace(
            experience=state_exp,
            current_index=current_index,
            is_full=is_full,
        )
    else:
        warnings.warn(
            "Given data is incompatible with the Buffer!\n" +
            f"Buffer fields: {', '.join(sorted(list(state.experience.keys())))}\n" +
            f"Data fields: {', '.join(sorted(list(stored_state.experience.keys())))}"
        )
    return state

def save_object_attr_pickle(obj, attrs, dir_path):
    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True, parents=True)
    for attr in attrs:
        attr_value = getattr(obj, attr)
        if hasattr(attr_value, "save"):
            attr_value.save(dir_path / attr)
        else:
            save_pickle(attr_value, dir_path / f"{attr}.pickle")

def load_object_attr_pickle(obj, attrs, dir_path):
    dir_path = Path(dir_path)
    attr_to_value, loaded_attrs = {}, {}
    for attr in attrs:
        value = getattr(obj, attr)
        if hasattr(value, "load"):
            load_dir = dir_path / attr
            if load_dir.exists():
                value, loaded_subattrs = value.load(load_dir)
                attr_to_value[attr] = value
                loaded_attrs[attr] = loaded_subattrs
            else:
                loaded_attrs[attr] = "-"
        else:
            load_path = dir_path / f"{attr}.pickle"
            if load_path.exists():
                value = load_pickle(load_path)
                attr_to_value[attr] = value
                loaded_attrs[attr] = "+"
            else:
                loaded_attrs[attr] = "-"
    return attr_to_value, loaded_attrs

class SaveLoadMixin:
    def save(self, dir_path: str) -> None:
        save_object_attr_pickle(self, self._save_attrs, dir_path)

    def load(self, dir_path: str) -> None:
        attr_to_value, loaded_attrs = load_object_attr_pickle(self, self._save_attrs, dir_path)
        for attr, value in attr_to_value.items():
            setattr(self, attr, value)
        return self, loaded_attrs

class SaveLoadFrozenDataclassMixin(SaveLoadMixin):
    def load(self, dir_path: str) -> None:
        attr_to_value, loaded_attrs = load_object_attr_pickle(self, self._save_attrs, dir_path)
        self = self.replace(**attr_to_value)
        return self, loaded_attrs

def get_buffer_state_size(buffer_state: BufferState) -> int:
    if buffer_state.is_full:
        key = list(buffer_state.experience.keys())[0]
        size = buffer_state.experience[key].shape[1]
    else:
        size = buffer_state.current_index
    return int(size)

def instantiate_jitted_fbx_buffer(fbx_buffer_config: DictConfig):
    buffer = instantiate(fbx_buffer_config)
    buffer = buffer.replace(
        init = jax.jit(buffer.init),
        add = jax.jit(buffer.add, donate_argnums=0),
        sample = jax.jit(buffer.sample),
        can_sample = jax.jit(buffer.can_sample),
    )
    return buffer

def convert_figure_to_array(figure: plt.Figure) -> np.ndarray:
    agg = figure.canvas.switch_backends(FigureCanvasAgg)
    agg.draw()
    return np.asarray(agg.buffer_rgba())

def get_method_partially(path: str, params: Dict):
    method = hydra.utils.get_method(path)
    method = functools.partial(method, **params)
    return method

@functools.partial(jax.jit, static_argnums=1)
def sample_batch(rng: PRNGKey, buffer: Buffer, state: BufferState):
    new_rng, key = jax.random.split(rng)
    batch = buffer.sample(state, key).experience
    return new_rng, batch
