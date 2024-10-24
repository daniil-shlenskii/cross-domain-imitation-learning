import warnings

import json

from utils.types import Buffer


def save_buffer(state: Buffer, path: str):
    with open(path, "w") as file:
        json.dump(state.experience, file)

def load_buffer(state: Buffer, path: str):
    with open(path) as file:
        experience = json.load(file)
    if state.experience.keys() == experience.keys():
        for k, v in experience.items():
            state.experience[k] = v
    else:
        warnings.warn(
            "Given data is incompatible with the Buffer!\n" +
            f"Buffer fiels: {', '.join(sorted(list(state.experience.key())))}\n" +
            f"Data fiels: {', '.join(sorted(list(experience.key())))}"
        )

    return state