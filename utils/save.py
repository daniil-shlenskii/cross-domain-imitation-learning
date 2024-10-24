import warnings

import json

from utils.types import Buffer


def save_buffer(state: Buffer, path: str):
    data_dict = {k: v for k, v in state.items()}
    with open(path, "w") as file:
        json.dump(data_dict, file)

def load_buffer(state: Buffer, path: str):
    with open(path) as file:
        data_dict = json.load(file)
    if state["experience"].keys() == data_dict["experience"].keys():
        for k, v in data_dict["experience"].items():
            state["experience"][k] = v
    else:
        warnings.warn(
            "Given data is incompatible with the Buffer!\n" +
            f"Buffer fields: {', '.join(sorted(list(state["experience"].keys())))}\n" +
            f"Data fields: {', '.join(sorted(list(data_dict["experience"].keys())))}"
        )

    return state