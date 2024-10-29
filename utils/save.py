import warnings

import pickle

from utils.types import Buffer


def save_buffer(state: Buffer, path: str):
    with open(path, "wb") as file:
        pickle.dump(state, file)

def load_buffer(state: Buffer, path: str):
    with open(path, "rb") as file:
        stored_state = pickle.load(file)
    if state.experience.keys() == stored_state.experience.keys():
        state = stored_state
    else:
        warnings.warn(
            "Given data is incompatible with the Buffer!\n" +
            f"Buffer fields: {', '.join(sorted(list(state.experience.keys())))}\n" +
            f"Data fields: {', '.join(sorted(list(stored_state.experience.keys())))}"
        )
    return state