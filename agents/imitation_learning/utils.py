import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils.types import DataType


def get_state_pairs(batch: DataType):
    return jnp.concatenate([
        batch["observations"],
        batch["observations_next"],
    ], axis=1)


def get_state_and_policy_tsne_scatterplots(
    imitation_agent: "ImitationAgent",
    seed: int,
    learner_trajs: DataType,
):
    observation_keys = ["observations", "observations_next"]

    # get trajectories
    end_of_firt_traj_idx = np.argmax(learner_trajs["dones"])
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
