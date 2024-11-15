import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Tuple

import jax
import optax
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from utils.types import Buffer


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

def load_buffer(state: Buffer, path: str):
    stored_state = load_pickle(path)
    if state.experience.keys() == stored_state.experience.keys():
        state = stored_state
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
                # loaded_attrs.append(attr: loaded_subattrs)
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
    
def make_jitted_fbx_buffer(fbx_buffer_config: DictConfig):
    buffer = instantiate(fbx_buffer_config)
    buffer = buffer.replace(
        init = jax.jit(buffer.init),
        add = jax.jit(buffer.add, donate_argnums=0),
        sample = jax.jit(buffer.sample),
        can_sample = jax.jit(buffer.can_sample),
    )
    return buffer