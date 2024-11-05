import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import optax
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
    for attr in attrs:
        attr_value = getattr(obj, attr)
        if hasattr(attr_value, "save"):
            attr_value.save(dir_path / attr)
        else:
            load_pickle(attr_value, dir_path / f"{attr}.pickle")

def load_object_attr_pickle(obj, attrs, dir_path):
    dir_path = Path(dir_path)
    loaded_attrs = []
    for attr in attrs:
        attr_value = getattr(obj, attr)
        if hasattr(attr_value, "load"):
            load_dir = dir_path / attr
            if load_dir.exists():
                attr_value.load(load_dir)
                loaded_attrs.append(attr)
        else:
            load_path = dir_path / f"{attr}.pickle"
            if load_path.exists():
                attr_value.load(load_path)
                load_pickle(attr_value, load_path)
                loaded_attrs.append(attr)
    return loaded_attrs
