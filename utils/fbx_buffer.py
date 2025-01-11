import functools
import warnings

import jax
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from .custom_types import Buffer, BufferState, PRNGKey
from .save_and_load import load_pickle, save_pickle


@functools.partial(jax.jit, static_argnums=1)
def sample_batch_jit(rng: PRNGKey, buffer: Buffer, state: BufferState):
    new_rng, key = jax.random.split(rng)
    batch = buffer.sample(state, key).experience
    return new_rng, batch

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

def save_buffer(state: Buffer, path: str):
    save_pickle(state, path)
