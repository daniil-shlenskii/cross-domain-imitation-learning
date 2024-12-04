import functools
import math
from collections import defaultdict
from copy import deepcopy
from typing import Dict

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from sklearn.manifold import TSNE

from agents.base_agent import Agent
from gan.discriminator import Discriminator
from utils.evaluate import apply_model_jit, evaluate
from utils.types import Buffer, BufferState, PRNGKey
from utils.utils import get_buffer_state_size


def get_state_and_policy_tsne_scatterplots(
    seed: int,
    #
    dida_agent: Agent,
    env: gym.Env,
    num_episodes: int,
    #
    expert_buffer_state: BufferState,
    anchor_buffer_state: BufferState,
):
    observation_keys = ["observations", "observations_next"]

    # collect trajectories
    _, learner_trajs = evaluate(
        agent=dida_agent,
        env=env,
        num_episodes=num_episodes,
        seed=seed,
        return_trajectories=True
    )
    rollouts_size = learner_trajs["observations"].shape[0]
    expert_trajs = {k: expert_buffer_state.experience[k][0, :rollouts_size] for k in observation_keys}
    anchor_trajs = {k: anchor_buffer_state.experience[k][0, :rollouts_size] for k in observation_keys}
    
    # encode trjectories
    for k in observation_keys:
        learner_trajs[k] = encode_observation_jit(dida_agent.learner_encoder, learner_trajs[k])
        expert_trajs[k] = encode_observation_jit(dida_agent.expert_encoder, expert_trajs[k])
        anchor_trajs[k] = encode_observation_jit(dida_agent.expert_encoder, anchor_trajs[k])

    # state and policy embeddings
    learner_state_embeddings = learner_trajs["observations"]
    expert_state_embeddings = expert_trajs["observations"]

    learner_policy_embeddings = np.concatenate([learner_trajs["observations"], learner_trajs["observations_next"]], axis=1)
    expert_policy_embeddings = np.concatenate([expert_trajs["observations"], expert_trajs["observations_next"]], axis=1)
    anchor_policy_embeddings = np.concatenate([anchor_trajs["observations"], anchor_trajs["observations_next"]], axis=1)

    # combine embeddings for further processing
    state_embeddings_list = [
        learner_state_embeddings,
        expert_state_embeddings
    ]
    policy_embeddings_list = [
        learner_policy_embeddings,
        expert_policy_embeddings,
        anchor_policy_embeddings,
    ]

    state_size_list = [0] + list(np.cumsum(list(map(
        lambda embs: embs.shape[0],
        state_embeddings_list,
    ))))
    policy_size_list = [0] + list(np.cumsum(list(map(
        lambda embs: embs.shape[0],
        policy_embeddings_list,
    ))))

    # tsne embeddings
    tsne_state_embeddings = TSNE(random_state=seed).fit_transform(np.concatenate(state_embeddings_list))
    tsne_policy_embeddings = TSNE(random_state=seed).fit_transform(np.concatenate(policy_embeddings_list))

    tsne_state_embeddings_list = [
        tsne_state_embeddings[
            state_size_list[i]: state_size_list[i + 1]
        ]
        for i in range(len(state_embeddings_list))
    ]
    tsne_policy_embeddings_list = [
        tsne_policy_embeddings[
            policy_size_list[i]: policy_size_list[i + 1]
        ]
        for i in range(len(policy_embeddings_list))
    ]

    # scatterplots
    scatter_params_list = (
        {"label": "TE", "c": "tab:blue",   "marker": "x"},
        {"label": "SE", "c": "tab:red",    "marker": "o"},
        {"label": "SR", "c": "tab:orange", "marker": "s"},
    )
    figsize=(5, 5)
    
    state_figure = plt.figure(figsize=figsize)
    for tsne_state_embeddings, scatter_params in zip(tsne_state_embeddings_list, scatter_params_list):
        plt.scatter(tsne_state_embeddings[:, 0], tsne_state_embeddings[:, 1], **scatter_params)
    plt.legend()
    plt.close()

    policy_figure = plt.figure(figsize=figsize)
    for tsne_policy_embeddings, scatter_params in zip(tsne_policy_embeddings_list, scatter_params_list):
        plt.scatter(tsne_policy_embeddings[:, 0], tsne_policy_embeddings[:, 1], **scatter_params)
    plt.legend()
    plt.close()

    return state_figure, policy_figure

def get_discriminators_hists(
    seed: int,
    #
    dida_agent: Agent,
    env: gym.Env,
    #
    expert_buffer_state: BufferState
):
    observation_keys = ["observations", "observations_next"]

    # learner trajectory
    _, learner_traj = evaluate(
        agent=dida_agent,
        env=env,
        num_episodes=1,
        seed=seed,
        return_trajectories=True
    )

    # expert trajectory
    expert_exp = expert_buffer_state.experience
    end_of_first_traj = np.argmax(expert_exp["dones"][0])
    expert_traj = {k: expert_buffer_state.experience[k][0, :end_of_first_traj] for k in observation_keys}

    # encode trjectories
    for k in observation_keys:
        learner_traj[k] = encode_observation_jit(dida_agent.learner_encoder, learner_traj[k])
        expert_traj[k] = encode_observation_jit(dida_agent.expert_encoder, expert_traj[k])

    # state and policy embeddings
    learner_state_embeddings = learner_traj["observations"]
    expert_state_embeddings = expert_traj["observations"]

    learner_policy_embeddings = np.concatenate([learner_traj["observations"], learner_traj["observations_next"]], axis=1)
    expert_policy_embeddings = np.concatenate([expert_traj["observations"], expert_traj["observations_next"]], axis=1)

    # state and policy logits
    state_learner_logits = apply_model_jit(dida_agent.domain_discriminator, learner_state_embeddings)
    state_expert_logits = apply_model_jit(dida_agent.domain_discriminator, expert_state_embeddings)

    policy_learner_logits = apply_model_jit(dida_agent.policy_discriminator, learner_policy_embeddings)
    policy_expert_logits = apply_model_jit(dida_agent.policy_discriminator, expert_policy_embeddings)

    # plots
    figsize=(5, 5)

    state_learner_figure = plt.figure(figsize=figsize)
    plt.plot(state_learner_logits, "bo")
    plt.close()

    state_expert_figure = plt.figure(figsize=figsize)
    plt.plot(state_expert_logits, "bo")
    plt.close()

    policy_learner_figure = plt.figure(figsize=figsize)
    plt.plot(policy_learner_logits, "bo")
    plt.close()

    policy_expert_figure = plt.figure(figsize=figsize)
    plt.plot(policy_expert_logits, "bo")
    plt.close()

    return state_learner_figure, state_expert_figure, policy_learner_figure, policy_expert_figure
