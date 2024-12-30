from copy import deepcopy

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.manifold import TSNE

from utils import (get_buffer_state_size, instantiate_jitted_fbx_buffer,
                   load_pickle)
from utils.types import BufferState, DataType


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
        random_buffer_state.experience["observations_next"].at[0].set(
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

def get_state_and_policy_tsne_scatterplots(
    imitation_agent: "ImitationAgent",
    seed: int,
    learner_trajs: DataType,
):
    observation_keys = ["observations", "observations_next"]

    # get trajectories
    end_of_firt_traj_idx = np.argmax(learner_trajs["truncated"])
    learner_traj = {k: learner_trajs[k][:end_of_firt_traj_idx] for k in observation_keys}
    expert_traj = {k: imitation_agent.expert_buffer_state.experience[k][0, :end_of_firt_traj_idx] for k in observation_keys}

    # encode trjectories
    for k in observation_keys:
        learner_traj[k] = imitation_agent._preprocess_observations(learner_traj[k])
        expert_traj[k] = imitation_agent._preprocess_expert_observations(expert_traj[k])

    # state and state_pairs embeddings
    learner_state_embeddings = learner_traj["observations"]
    expert_state_embeddings = expert_traj["observations"]

    learner_state_pairs_embeddings = get_state_pairs(learner_traj)
    expert_state_pairs_embeddings = get_state_pairs(expert_traj)

    # combine embeddings for further processing
    state_embeddings_list = [
        learner_state_embeddings,
        expert_state_embeddings
    ]
    state_pairs_embeddings_list = [
        learner_state_pairs_embeddings,
        expert_state_pairs_embeddings,
    ]

    state_size_list = [0] + list(np.cumsum(list(map(
        lambda embs: embs.shape[0],
        state_embeddings_list,
    ))))
    state_pairs_size_list = [0] + list(np.cumsum(list(map(
        lambda embs: embs.shape[0],
        state_pairs_embeddings_list,
    ))))

    # tsne embeddings
    tsne_state_embeddings = TSNE(random_state=seed).fit_transform(np.concatenate(state_embeddings_list))
    tsne_state_pairs_embeddings = TSNE(random_state=seed).fit_transform(np.concatenate(state_pairs_embeddings_list))

    tsne_state_embeddings_list = [
        tsne_state_embeddings[
            state_size_list[i]: state_size_list[i + 1]
        ]
        for i in range(len(state_embeddings_list))
    ]
    tsne_state_pairs_embeddings_list = [
        tsne_state_pairs_embeddings[
            state_pairs_size_list[i]: state_pairs_size_list[i + 1]
        ]
        for i in range(len(state_pairs_embeddings_list))
    ]

    # scatterplots
    opaque = np.linspace(.2, 1., num=end_of_firt_traj_idx)
    scatter_params_list = (
        {"label": "TE", "c": "tab:blue",   "marker": "x", "alpha": opaque},
        {"label": "SE", "c": "tab:red",    "marker": "o", "alpha": opaque},
    )
    figsize=(5, 5)

    state_figure = plt.figure(figsize=figsize)
    for tsne_state_embeddings, scatter_params in zip(tsne_state_embeddings_list, scatter_params_list):
        plt.scatter(tsne_state_embeddings[:, 0], tsne_state_embeddings[:, 1], **scatter_params)
    plt.legend()
    plt.close()

    state_pairs_figure = plt.figure(figsize=figsize)
    for tsne_state_pairs_embeddings, scatter_params in zip(tsne_state_pairs_embeddings_list, scatter_params_list):
        plt.scatter(tsne_state_pairs_embeddings[:, 0], tsne_state_pairs_embeddings[:, 1], **scatter_params)
    plt.legend()
    plt.close()

    return state_figure, state_pairs_figure
