from copy import deepcopy

import jax.numpy as jnp
import numpy as np

from utils import (get_buffer_state_size, instantiate_jitted_fbx_buffer,
                   load_pickle)
from utils.custom_types import BufferState


class OTBufferFactory:
    def  __init__(
        self,
        anchor_types: list[str]=[],
        random_buffer_path: str=None,
        seed: int=0
    ):
        self.seed = seed
        self.anchor_types = anchor_types
        self.random_buffer_path = random_buffer_path
        self.type_to_func = {
            "reversed": get_reversed_state_dict,
            "shuffled": get_shuffled_state_dict,
            "stationary": get_stationary_state_dict,
            "noised": get_noised_state_dict,
            "interpolated": get_interpolated_state_dict,
        }

    def __call__(self, expert_state: BufferState, batch_size: int):
        # extract buffer dict from buffer state
        expert_state_size = get_buffer_state_size(expert_state)
        expert_state_dict = {
            k: v[0, :expert_state_size] for k, v in expert_state.experience.items()
        }

        # get ot state dict
        ot_state_dict = self.get_ot_state_dict(expert_state_dict)

        # init ot buffer
        ot_state_size = ot_state_dict["observations"].shape[0]
        ot_buffer = instantiate_jitted_fbx_buffer({
            "_target_": "flashbax.make_item_buffer",
            "sample_batch_size": batch_size,
            "min_length": 1,
            "max_length": ot_state_size,
            "add_batches": False,
        })

        # init ot state
        ot_state_init_sample = {k: v[0] for k, v in ot_state_dict.items()}
        ot_state = ot_buffer.init(ot_state_init_sample)

        # fill ot state with ot dict
        ot_state_experience = {k: v[None] for k, v in ot_state_dict.items()}
        ot_state = ot_state.replace(
            experience=ot_state_experience,
            current_index=0,
            is_full=True,
        )

        print(f"{get_buffer_state_size(ot_state) = }")
        return ot_buffer, ot_state

    def get_ot_state_dict(self, expert_state_dict: dict):
        state_dicts_list = [deepcopy(expert_state_dict)]
        if self.random_buffer_path is not None:
            random_buffer_state = load_pickle(self.random_buffer_path)
            random_buffer_state_dict = {
                k: v[0] for k, v in random_buffer_state.experience.items()
            }
            state_dicts_list.append(random_buffer_state_dict)
        for i, anchor_type in enumerate(self.anchor_types):
            state_dicts_list.append(
               self.type_to_func[anchor_type](expert_state_dict, seed=self.seed+i)
            )

        ot_state_dict = {}
        for k in expert_state_dict:
            ot_state_dict[k] = jnp.concatenate([d[k] for d in state_dicts_list])

        return ot_state_dict


def get_reversed_state_dict(expert_state_dict: dict, seed: int):
    reversed_state_dict = deepcopy(expert_state_dict)
    reversed_state_dict["observations"], reversed_state_dict["observations_next"] =\
        reversed_state_dict["observations_next"], reversed_state_dict["observations"]
    return reversed_state_dict

def get_shuffled_state_dict(expert_state_dict: dict, seed: int):
    np.random.seed(seed)
    shuffled_state_dict = deepcopy(expert_state_dict)

    dict_size = expert_state_dict["observations"].shape[0]
    obs_next_perm_idcs = np.random.choice(dict_size, size=dict_size)

    shuffled_state_dict["observations_next"] =\
        expert_state_dict["observations_next"][obs_next_perm_idcs]

    return shuffled_state_dict

def get_stationary_state_dict(expert_state_dict: dict, seed: int):
    stationary_state_dict = deepcopy(expert_state_dict)
    stationary_state_dict["observations_next"] = expert_state_dict["observations"]
    return stationary_state_dict

def get_noised_state_dict(expert_state_dict: dict, seed: int):
    np.random.seed(seed)
    noised_state_dict = deepcopy(expert_state_dict)
    shape = expert_state_dict["observations"].shape
    dists = np.linalg.norm(expert_state_dict["observations"] - expert_state_dict["observations_next"], axis=-1)
    sigmas = dists / 6.
    noised_state_dict["observations"] = expert_state_dict["observations"] + np.random.randn(*shape) * sigmas[:, None]
    noised_state_dict["observations_next"] = expert_state_dict["observations_next"] + np.random.randn(*shape) * sigmas[:, None]
    return noised_state_dict

def get_interpolated_state_dict(expert_state_dict: dict, seed: int):
    np.random.seed(seed)
    interpolated_state_dict = deepcopy(expert_state_dict)

    alpha = np.random.rand() / 2.
    alpha_next = np.random.rand() / 2.
    interpolated_state_dict["observations"] = expert_state_dict["observations"] * (1 - alpha) + expert_state_dict["observations_next"] * alpha
    interpolated_state_dict["observations_next"] = expert_state_dict["observations_next"] * (1 - alpha_next) + expert_state_dict["observations"] * alpha_next
    return interpolated_state_dict
