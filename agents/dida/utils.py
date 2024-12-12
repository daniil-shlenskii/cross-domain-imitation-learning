import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from agents.base_agent import Agent
from agents.gail.utils import get_state_pairs
from utils import apply_model_jit
from utils.types import DataType

MIN_TRAJECTORY_SIZE = 100

def get_state_and_policy_tsne_scatterplots(
    dida_agent: Agent,
    seed: int,
    learner_trajs: DataType,
):
    observation_keys = ["observations", "observations_next"]

    # get trajectories
    end_of_firt_traj_idx = np.argmax(learner_trajs["dones"])
    end_of_firt_traj_idx = max(end_of_firt_traj_idx, MIN_TRAJECTORY_SIZE)
    learner_traj = {k: learner_trajs[k][:end_of_firt_traj_idx] for k in observation_keys}
    expert_traj = {k: dida_agent.expert_buffer_state.experience[k][0, :end_of_firt_traj_idx] for k in observation_keys}
    anchor_traj = {k: dida_agent.anchor_buffer_state.experience[k][0, :end_of_firt_traj_idx] for k in observation_keys}

    # encode trjectories
    for k in observation_keys:
        learner_traj[k] = dida_agent._preprocess_observations(learner_traj[k])
        expert_traj[k] = dida_agent._preprocess_expert_observations(expert_traj[k])
        anchor_traj[k] = dida_agent._preprocess_expert_observations(anchor_traj[k])

    # state and state_pairs embeddings
    learner_state_embeddings = learner_traj["observations"]
    expert_state_embeddings = expert_traj["observations"]

    learner_state_pairs_embeddings = get_state_pairs(learner_traj)
    expert_state_pairs_embeddings = get_state_pairs(expert_traj)
    anchor_state_pairs_embeddings = get_state_pairs(anchor_traj)

    # combine embeddings for further processing
    state_embeddings_list = [
        learner_state_embeddings,
        expert_state_embeddings
    ]
    state_pairs_embeddings_list = [
        learner_state_pairs_embeddings,
        expert_state_pairs_embeddings,
        anchor_state_pairs_embeddings,
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
    opaque = np.linspace(0, 1, num=end_of_firt_traj_idx)
    scatter_params_list = (
        {"label": "TE", "c": "tab:blue",   "marker": "x", "alpha": opaque},
        {"label": "SE", "c": "tab:red",    "marker": "o", "alpha": opaque},
        {"label": "SR", "c": "tab:orange", "marker": "s"},
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

def get_discriminators_hists(
    dida_agent: Agent,
    learner_trajs: DataType,
):
    observation_keys = ["observations", "observations_next"]

    # get trajectories
    end_of_firt_traj_idx = np.argmax(learner_trajs["dones"])
    end_of_firt_traj_idx = max(end_of_firt_traj_idx, MIN_TRAJECTORY_SIZE)
    learner_traj = {k: learner_trajs[k][:end_of_firt_traj_idx] for k in observation_keys}
    expert_traj = {k: dida_agent.expert_buffer_state.experience[k][0, :end_of_firt_traj_idx] for k in observation_keys}

    # encode trjectories
    for k in observation_keys:
        learner_traj[k] = dida_agent._preprocess_observations(learner_traj[k])
        expert_traj[k] = dida_agent._preprocess_expert_observations(expert_traj[k])

    # state_pairs embeddings
    learner_state_pairs_embeddings = get_state_pairs(learner_traj)
    expert_state_pairs_embeddings = get_state_pairs(expert_traj)

    # state and state_pairs logits
    state_discriminator = dida_agent.domain_encoder.state_discriminator
    state_learner_logits = apply_model_jit(state_discriminator, learner_state_pairs_embeddings)
    state_expert_logits = apply_model_jit(state_discriminator, expert_state_pairs_embeddings)

    policy_discriminator = dida_agent.domain_encoder.policy_discriminator
    policy_learner_logits = apply_model_jit(policy_discriminator, learner_state_pairs_embeddings)
    policy_expert_logits = apply_model_jit(policy_discriminator, expert_state_pairs_embeddings)

    # plots
    def logits_to_plot(logits):
        figure = plt.figure(figsize=(5, 5))
        plt.plot(logits, "bo")
        plt.axhline(y=0., color='r', linestyle='-')
        plt.close()
        return figure

    state_learner_figure = logits_to_plot(state_learner_logits)
    state_expert_figure = logits_to_plot(state_expert_logits)
    policy_learner_figure = logits_to_plot(policy_learner_logits)
    policy_expert_figure = logits_to_plot(policy_expert_logits)

    return state_learner_figure, state_expert_figure, policy_learner_figure, policy_expert_figure
