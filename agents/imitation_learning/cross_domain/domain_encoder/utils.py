from copy import deepcopy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from agents.imitation_learning.utils import (TRAJECTORIES_SCATTER_PARAMS,
                                             get_state_pairs)
from misc.gan.discriminator import LoosyDiscriminator
from nn.train_state import TrainState
from utils import cosine_similarity_fn, project_a_to_b
from utils.custom_types import DataType, Params
from utils.math import scalar_product_fn


@jax.jit
def encode_batch(encoder_state: TrainState, batch: DataType):
    batch["observations"], batch["observations_next"] = encode_states_given_params(
        encoder_state.params, encoder_state, batch["observations"], batch["observations_next"]
    )
    return batch

@jax.jit
def encode_states_given_params(params: Params, state: TrainState, states: jnp.ndarray, states_next: jnp.ndarray):
    batch_size = states.shape[0]
    observations = jnp.concatenate([states, states_next])
    encoded_observations = state.apply_fn({"params": params}, observations)
    return encoded_observations.at[:batch_size].get(), encoded_observations.at[batch_size:].get()


##### Divergence Scores #####

def get_discriminators_divergence_scores(*, domain_encoder: "BaseDomainEncoder", seed: int=0):
    info_key_prefix = "domain_encoder"

    # sample batches
    rng = jax.random.key(seed)
    rng, target_random_batch, source_random_batch, source_expert_batch = domain_encoder.sample_batches(rng)

    # source expert
    source_expert_divergence_scores = get_divergence_scores_dict(
        domain_encoder=domain_encoder,
        batch=source_expert_batch,
        is_target=False,
    )
    source_expert_divergence_scores = {f"{info_key_prefix}/source_expert/{k}": v for k, v in source_expert_divergence_scores.items()}

    divergence_scores = {**source_expert_divergence_scores}
    return divergence_scores

def flatten_fn(params_dict: dict):
    return jnp.concatenate([
        jnp.ravel(x) for x in
        jax.tree.flatten(params_dict, is_leaf=lambda x: isinstance(x, jnp.ndarray))[0]
    ])

def get_grad_wrt_params(
    *, domain_encoder, batch, state, state_loss, policy_loss, 
):
    # state grad
    state_grad = jax.value_and_grad(state_loss, has_aux=True)(
        state.params,
        state=state,
        discriminator=domain_encoder.discriminators.state_discriminator,
        states=batch["observations"],
     )[1]
    state_grad = flatten_fn(state_grad)

    # policy grad
    (_, info), policy_grad = jax.value_and_grad(policy_loss, has_aux=True)(
        state.params,
        state=state,
        discriminator=domain_encoder.discriminators.policy_discriminator,
        states=batch["observations"],
        states_next=batch["observations_next"],
    )
    policy_grad = flatten_fn(policy_grad)

    encoded_batch = deepcopy(batch)
    encoded_batch["observations"] = info["states"]
    encoded_batch["observations_next"] = info["states_next"]

    return state_grad, policy_grad, encoded_batch

def get_grad_wrt_embs(
    *, domain_encoder, encoded_batch, state_loss, policy_loss
):
    # state grad
    state_grad = jax.grad(state_loss)(
        encoded_batch["observations"],
        discriminator=domain_encoder.discriminators.state_discriminator
    ).mean(0)

    # policy grad
    policy_grad = jax.grad(policy_loss)(
        encoded_batch["observations"],
        states_next=encoded_batch["observations_next"],
        discriminator=domain_encoder.discriminators.policy_discriminator
    ).mean(0)

    return state_grad, policy_grad

def divergence_scores_fn(*, to_be_projected: jnp.ndarray, project_to: jnp.ndarray):
    projection = project_a_to_b(a=to_be_projected, b=project_to)

    projection_norm = jnp.linalg.norm(projection)
    project_to_norm = jnp.linalg.norm(project_to)

    if projection_norm == 0.:
        s = 1.
    else:
        s = jnp.sign(cosine_similarity_fn(projection, project_to))

    divergence_score = s * projection_norm / project_to_norm

    return divergence_score

def get_divergence_scores_dict(
    *, batch, domain_encoder, is_target
):
    divergence_scores = {}

    # preparation

    if is_target:
        state = domain_encoder.target_encoder.state
        params_state_loss = state.loss_fn.state_real_loss_given_params
        params_policy_loss = state.loss_fn.policy_fake_loss_given_params
        embs_state_loss = state.loss_fn.state_real_loss
        embs_policy_loss = state.loss_fn.policy_fake_loss
    else:
        state = domain_encoder.source_encoder.state
        params_state_loss = state.loss_fn.state_fake_loss_given_params
        params_policy_loss = state.loss_fn.policy_real_loss_given_params
        embs_state_loss = state.loss_fn.state_fake_loss
        embs_policy_loss = state.loss_fn.policy_real_loss

    # divergence scores wrt params
    state_grad_wrt_params, policy_grad_wrt_params, encoded_batch = get_grad_wrt_params(
        domain_encoder=domain_encoder,
        state=state,
        batch=batch,
        state_loss=params_state_loss,
        policy_loss=params_policy_loss,
    )

    divergence_scores["divergence_score_wrt_params/state_to_policy"] =\
        divergence_scores_fn(to_be_projected=state_grad_wrt_params, project_to=policy_grad_wrt_params)

    divergence_scores["divergence_score_wrt_params/policy_to_state"] =\
        divergence_scores_fn(to_be_projected=policy_grad_wrt_params, project_to=state_grad_wrt_params)

    divergence_scores["cos_sim_wrt_params"] =\
        cosine_similarity_fn(policy_grad_wrt_params, state_grad_wrt_params)

    # divergence scores wrt embs

    state_grad_wrt_embs, policy_grad_wrt_embs = get_grad_wrt_embs(
        domain_encoder=domain_encoder,
        encoded_batch=encoded_batch,
        state_loss=embs_state_loss,
        policy_loss=embs_policy_loss,
    )

    divergence_scores["divergence_score_wrt_embs/state_to_policy"] =\
        divergence_scores_fn(to_be_projected=state_grad_wrt_embs, project_to=policy_grad_wrt_embs)

    divergence_scores["divergence_score_wrt_embs/policy_to_state"] =\
        divergence_scores_fn(to_be_projected=policy_grad_wrt_embs, project_to=state_grad_wrt_embs)

    divergence_scores["cos_sim_wrt_embs"] =\
        cosine_similarity_fn(policy_grad_wrt_embs, state_grad_wrt_embs)

    return divergence_scores

##### Two Dim Data Plot #####

def get_two_dim_data_plot(*, traj_dict: dict, state_discriminator: LoosyDiscriminator):
    # get state discriminator hyperplane
    state_discr_params = state_discriminator.state.params
    b, n = jax.tree.flatten(state_discr_params)[0][-2:]
    assert n.shape[-1] == 1, n.ndim == 2
    b, n = b.squeeze(-1), n.squeeze(-1)
    b /= jnp.linalg.norm(n)
    n /= jnp.linalg.norm(n)

    ## get a and x0
    def get_hyperplane_a_and_x0(n, b):
        a = np.zeros_like(n)
        a[0] = n[1]
        a[1] = -n[0]

        if n[0] != 0.:
            c = -b / n[0]
            x0 = np.array([c, 0.])
        else:
            c = -b / n[1]
            x0 = np.array([0., c])

        return a, x0

    a, x0 = get_hyperplane_a_and_x0(n, b)
    assert (
        np.isclose((a * n).sum(), 0., atol=1e-4) and
        np.isclose((x0 * n).sum(), -b, atol=1e-4)
    ), f"{(a * n).sum() = } and {(x0 * n).sum() = },and {b = }"

    ## project mean of trajectories to the hyperplane
    traj_mean = (traj_dict["TR"].mean(0) + traj_dict["SE"].mean(0)) * 0.5
    traj_mean_proj = project_a_to_b(traj_mean - x0, a)
    x0 = x0 + traj_mean_proj
    assert np.isclose(scalar_product_fn(n, x0) + b, 0, atol=1e-4), f"{scalar_product_fn(n, x0) + b = }"

    ## two line's points
    span = jnp.abs(traj_dict["TR"].mean(0) - traj_dict["SE"].mean(0))
    h1 = x0 + a * span * 0.5
    h2 = x0 - a * span * 0.5

    ## normal line's points
    n1 = x0
    n2 = x0 + n * span * 0.1

    # plot
    figsize=(5, 5)
    figure = plt.figure(figsize=figsize)

    plt.scatter(traj_dict["TR"][:, 0], traj_dict["TR"][:, 1], **TRAJECTORIES_SCATTER_PARAMS["TR"])
    plt.scatter(traj_dict["SE"][:, 0], traj_dict["SE"][:, 1], **TRAJECTORIES_SCATTER_PARAMS["SE"])
    plt.plot([h1[0], h2[0]], [h1[1], h2[1]], color="k")
    plt.plot([n1[0], n2[0]], [n1[1], n2[1]], color="k")
    plt.close()

    return figure
