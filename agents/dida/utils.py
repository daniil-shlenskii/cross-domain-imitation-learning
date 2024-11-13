import functools
import math
from collections import defaultdict
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from frozendict import frozendict
from sklearn.manifold import TSNE

from utils.types import Buffer, BufferState


@jax.jit
def process_policy_discriminator_input(x: jnp.ndarray):
    doubled_b_size, dim = x.shape
    x = x.reshape(2, doubled_b_size // 2, dim).transpose(1, 0, 2).reshape(-1, dim * 2)
    return x

@jax.jit
def encode_observation(encoder, observations):
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
    # prepare learner, expert, anchor data info
    buffers_tpl = (learner_buffer, expert_buffer, expert_buffer)
    buffer_states_tpl = (learner_buffer_state, expert_buffer_state, anchor_buffer_state)
    encoders_tpl = (learner_encoder, expert_encoder, expert_encoder)
    scatter_params_tpl = (
        {"label": "learner", "c": "tab:blue",   "marker": "x"},
        {"label": "expert",  "c": "tab:red",    "marker": "o"},
        {"label": "anchor",  "c": "tab:orange", "marker": "s"},
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
        embeddings = [[], [], []]
        
        rng = jax.random.key(seed)
        for i, (buffer, buffer_state, encoder, n_iters) in enumerate(zip(
            buffers_tpl, buffer_states_tpl, encoders_tpl, n_iters_tpl
        )):
            for _ in range(n_iters):
                rng, key = jax.random.split(rng)
                batch = buffer.sample(buffer_state, key).experience
                encoded_obsevations = encoder(batch["observations"]) 
                encoded_obsevations_next = encoder(batch["observations_next"]) 
                encoded_behavior = jnp.concatenate([encoded_obsevations, encoded_obsevations_next], axis=1)
                embeddings[i].append(encoded_behavior)
            embeddings[i] = jnp.concatenate(embeddings[i])
        return embeddings

    # behavior embeddings
    behavior_embeddings_tpl = compute_embeddings(seed, buffers_tpl, buffer_states_tpl, encoders_tpl, n_iters_tpl)
    behavior_embeddings_tpl = tuple(behavior_emb[:n_samples_per_buffer] for behavior_emb in behavior_embeddings_tpl)

    # get tsne embeddings
    tsne_embeddings = TSNE(random_state=seed).fit_transform(np.concatenate(behavior_embeddings_tpl))
    tsne_embeddings_tpl = tuple(
        tsne_embeddings[i * n_samples_per_buffer: (i + 1) * n_samples_per_buffer]
        for i in range(3)
    )
    
    # get scatterplot figure
    fig = plt.figure(figsize=(4, 4)) 
    for tsne_embeddings, scatter_params in zip(tsne_embeddings_tpl, scatter_params_tpl):
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], **scatter_params)
    plt.legend()

    return fig
