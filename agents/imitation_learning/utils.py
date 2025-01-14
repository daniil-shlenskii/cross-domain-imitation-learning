from copy import deepcopy
from typing import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.manifold import TSNE

from utils import (get_buffer_state_size, instantiate_jitted_fbx_buffer,
                   load_pickle)
from utils.custom_types import BufferState, DataType


def get_state_pairs(batch: DataType):
    return jnp.concatenate([
        batch["observations"],
        batch["observations_next"],
    ], axis=1)

def get_random_from_expert_buffer_state(*, seed: int, expert_buffer_state: BufferState):
    buffer_state_size = get_buffer_state_size(expert_buffer_state)
    random_buffer_state = deepcopy(expert_buffer_state)

    np.random.seed(seed)
    obs_next_perm_idcs = np.random.choice(buffer_state_size)

    random_buffer_state.experience["observations_next"] = \
        random_buffer_state.experience["observations_next"].at[0, :buffer_state_size].set(
            random_buffer_state.experience["observations_next"][0, obs_next_perm_idcs]
        )

    return random_buffer_state

def prepare_buffer(
    buffer_state_path: str,
    batch_size: int,
    buffer_state_processor_config: DictConfig = None,
):
    # load buffer state
    _buffer_state = load_pickle(buffer_state_path)

    # buffer init
    buffer_state_size = get_buffer_state_size(_buffer_state)
    buffer = instantiate_jitted_fbx_buffer({
        "_target_": "flashbax.make_item_buffer",
        "sample_batch_size": batch_size,
        "min_length": 1,
        "max_length": buffer_state_size,
        "add_batches": False,
    })

    # remove dummy experience samples (flashbax specific stuff)
    buffer_state_exp = _buffer_state.experience
    buffer_state_init_sample = {k: v[0, 0] for k, v in buffer_state_exp.items()}
    buffer_state = buffer.init(buffer_state_init_sample)

    new_buffer_state_exp = {}
    for k, v in buffer_state_exp.items():
        new_buffer_state_exp[k] = jnp.asarray(v[0, :buffer_state_size][None])

    buffer_state = buffer_state.replace(
        experience=new_buffer_state_exp,
        current_index=0,
        is_full=True,
    )

    # process buffer state
    if buffer_state_processor_config is not None:
        buffer_state_processor = instantiate(buffer_state_processor_config)
        buffer_state = buffer_state_processor(buffer_state)

    return buffer, buffer_state

##### Visualization #####

FIGSIZE = (5, 5)

TRAJECTORIES_SCATTER_PARAMS = {
    "TR": {"label": "TR", "c": "tab:green",  "marker": "+"},
    "TE": {"label": "TE", "c": "tab:blue",   "marker": "x"},
    "SR": {"label": "SR", "c": "tab:orange", "marker": "o"},
    "SE": {"label": "SE", "c": "tab:red",    "marker": "s"},
}

def get_trajectory_from_buffer(state: BufferState):
    trajs = state.experience
    end_of_first_traj_idx = np.argmax(trajs["truncated"][0])
    assert end_of_first_traj_idx > 0
    traj = {k: trajs[k][0, :end_of_first_traj_idx] for k in trajs}
    return traj

def get_trajectory_from_dict(trajs: dict):
    end_of_first_traj_idx = np.argmax(trajs["truncated"])
    assert end_of_first_traj_idx > 0
    traj = {k: trajs[k][:end_of_first_traj_idx] for k in trajs}
    return traj

def get_trajs_tsne_scatterplot(
    *,
    traj_dict: dict,
    keys_to_use: Sequence,
    seed: int = 0,
):
    keys_to_use = list(keys_to_use)
    trajs_list = [traj_dict[k] for k in keys_to_use]

    # tsne embeddings
    tsne_embs = np.concatenate(trajs_list)

    dim = traj_dict[keys_to_use[0]].shape[-1]
    if dim > 2:
        tsne_embs = TSNE(random_state=seed).fit_transform(np.concatenate(trajs_list))
    elif dim == 1:
            # prepare fake axis to obtain two dim data
            fake_axis = []
            eps = 0.01
            for i, traj in enumerate(trajs_list):
                fake_axis.append(np.zeros_like(traj) + eps * i)
            fake_axis = np.concatenate(fake_axis)

            tsne_embs = np.concatenate([tsne_embs, fake_axis], axis=-1)
    else:
        pass

    # split tsne embs
    tsne_traj_dict = {}
    start_idx = 0
    for k, traj in zip(keys_to_use, trajs_list):
        end_idx = start_idx + len(traj)
        tsne_traj_dict[k] = tsne_embs[start_idx: end_idx]
        start_idx = end_idx


    figure = plt.figure(figsize=FIGSIZE)
    for k, tsne_embs in tsne_traj_dict.items():
        plt.scatter(tsne_embs[:, 0], tsne_embs[:, 1], **TRAJECTORIES_SCATTER_PARAMS[k])
    plt.legend()
    plt.close()

    return figure
