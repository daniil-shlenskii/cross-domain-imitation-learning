from typing import Any

import warnings

import pickle

from utils.types import Buffer


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