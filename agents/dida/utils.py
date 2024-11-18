import functools
import math
from collections import defaultdict
from copy import deepcopy
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils.types import Buffer, BufferState
from utils.utils import get_buffer_state_size


@jax.jit
def process_policy_discriminator_input(x: jnp.ndarray):
    doubled_b_size, dim = x.shape
    x = x.reshape(2, doubled_b_size // 2, dim).transpose(1, 0, 2).reshape(-1, dim * 2)
    return x

@jax.jit
def encode_observation_jit(encoder, observations):
    return encoder(observations)

def get_tsne_embeddings_scatter(
    *,
    seed: int,
    learner_buffer: Buffer,
    expert_buffer: Buffer,
    learner_buffer_state: BufferState,
    expert_buffer_state: BufferState,
    anchor_buffer_state: BufferState,
    learner_encoder: ...,
    expert_encoder: ...,
    n_samples_per_buffer: int,
):
    # prepare learner_random buffer state
    learner_random_buffer_state = deepcopy(learner_buffer_state)
    buffer_state_size = get_buffer_state_size(learner_random_buffer_state)
    perm_idcs = np.random.choice(buffer_state_size)
    learner_random_buffer_state.experience["observations_next"] = \
        learner_random_buffer_state.experience["observations_next"].at[0, :buffer_state_size].set(
            learner_random_buffer_state.experience["observations_next"][0, perm_idcs]
        )

    # prepare learner_random, learner, expert_random (aka anchor), expert data info
    buffers_tpl = (learner_buffer, learner_buffer, expert_buffer, expert_buffer)
    buffer_states_tpl = (learner_random_buffer_state, learner_buffer_state, anchor_buffer_state, expert_buffer_state)
    encoders_tpl = (learner_encoder, learner_encoder, expert_encoder, expert_encoder)
    scatter_params_tpl = (
        {"label": "TR", "c": "tab:green",  "marker": "*"},
        {"label": "TE", "c": "tab:blue",   "marker": "x"},
        {"label": "SR", "c": "tab:orange", "marker": "s"},
        {"label": "SE", "c": "tab:red",    "marker": "o"},
    )
    n_iters_tpl = []
    for buffer, buffer_state in zip(buffers_tpl, buffer_states_tpl):
        batch = buffer.sample(buffer_state, jax.random.key(0)).experience
        batch_size = batch["observations"].shape[0]
        n_iters_tpl.append(math.ceil(n_samples_per_buffer / batch_size))
    n_iters_tpl = tuple(n_iters_tpl)

    # collect data and encode it
    @functools.partial(jax.jit, static_argnames=("buffers_tpl", "n_iters_tpl"))
    def compute_embeddings(seed, buffers_tpl, buffer_states_tpl, encoders_tpl, n_iters_tpl):
        state_embeddings, behavior_embeddings = [[], [], [], []], [[], [], [], []]
        
        rng = jax.random.key(seed)
        for i, (buffer, buffer_state, encoder, n_iters) in enumerate(zip(
            buffers_tpl, buffer_states_tpl, encoders_tpl, n_iters_tpl
        )):
            for _ in range(n_iters):
                rng, key = jax.random.split(rng)
                batch = buffer.sample(buffer_state, key).experience
                encoded_observations = encoder(batch["observations"]) 
                encoded_observations_next = encoder(batch["observations_next"]) 
                encoded_behavior = jnp.concatenate([encoded_observations, encoded_observations_next], axis=1)

                state_embeddings[i].append(encoded_observations)
                behavior_embeddings[i].append(encoded_behavior)

            state_embeddings[i] = jnp.concatenate(state_embeddings[i])
            behavior_embeddings[i] = jnp.concatenate(behavior_embeddings[i])

        return state_embeddings, behavior_embeddings

    # get embeddings
    state_embeddings_tpl, behavior_embeddings_tpl = compute_embeddings(seed, buffers_tpl, buffer_states_tpl, encoders_tpl, n_iters_tpl)

    state_embeddings_tpl = tuple(state_emb[:n_samples_per_buffer] for state_emb in state_embeddings_tpl)
    behavior_embeddings_tpl = tuple(behavior_emb[:n_samples_per_buffer] for behavior_emb in behavior_embeddings_tpl)

    # get tsne embeddings
    tsne_state_embeddings = TSNE(random_state=seed).fit_transform(np.concatenate(state_embeddings_tpl))
    tsne_state_embeddings_tpl = tuple(
        tsne_state_embeddings[i * n_samples_per_buffer: (i + 1) * n_samples_per_buffer]
        for i in range(len(buffers_tpl))
    )
    
    tsne_behavior_embeddings = TSNE(random_state=seed).fit_transform(np.concatenate(behavior_embeddings_tpl))
    tsne_behavior_embeddings_tpl = tuple(
        tsne_behavior_embeddings[i * n_samples_per_buffer: (i + 1) * n_samples_per_buffer]
        for i in range(len(buffers_tpl))
    )
    
    # get scatterplot figure
    state_figure = plt.figure(figsize=(4, 4)) 
    for i, (tsne_state_embeddings, scatter_params) in enumerate(zip(tsne_state_embeddings_tpl, scatter_params_tpl)):
        if i in {1, 3}:
            plt.scatter(tsne_state_embeddings[:, 0], tsne_state_embeddings[:, 1], **scatter_params)
    plt.legend()
    plt.close()

    behavior_figure = plt.figure(figsize=(4, 4)) 
    for tsne_behavior_embeddings, scatter_params in zip(tsne_behavior_embeddings_tpl, scatter_params_tpl):
        plt.scatter(tsne_behavior_embeddings[:, 0], tsne_behavior_embeddings[:, 1], **scatter_params)
    plt.legend()
    plt.close()

    return state_figure, behavior_figure
