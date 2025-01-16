from copy import deepcopy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from agents.imitation_learning.utils import TRAJECTORIES_SCATTER_PARAMS
from misc.gan.discriminator import LoosyDiscriminator
from utils import cosine_similarity_fn, project_a_to_b

##### Divergence Scores #####

def get_discriminators_divergence_scores(*, domain_encoder: "BaseDomainEncoder", seed: int=0):
    info_key_prefix = "domain_encoder"

    # sample batches
    rng = jax.random.key(seed)
    rng, target_random_batch, source_random_batch, source_expert_batch = domain_encoder.sample_batches(rng)

    #
    state_discriminator = domain_encoder.discriminators.state_discriminator
    policy_discriminator = domain_encoder.discriminators.policy_discriminator

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

    if is_target:
        state = domain_encoder.target_encoder.state
        state_loss = state.loss_fn.target_state_loss
        policy_loss = state.loss_fn.target_random_policy_loss
    else:
        state = domain_encoder.source_encoder.state
        state_loss = state.loss_fn.source_state_loss
        policy_loss = state.loss_fn.source_expert_policy_loss

    state_grad_wrt_params, policy_grad_wrt_params, encoded_batch = get_grad_wrt_params(
        domain_encoder=domain_encoder,
        state=state,
        batch=batch,
        state_loss=state_loss,
        policy_loss=policy_loss,
    )

    divergence_scores["divergence_score_wrt_params/state_to_policy"] =\
        divergence_scores_fn(to_be_projected=state_grad_wrt_params, project_to=policy_grad_wrt_params)

    divergence_scores["divergence_score_wrt_params/policy_to_state"] =\
        divergence_scores_fn(to_be_projected=policy_grad_wrt_params, project_to=state_grad_wrt_params)

    divergence_scores["cos_sim_wrt_params"] =\
        cosine_similarity_fn(policy_grad_wrt_params, state_grad_wrt_params)

    return divergence_scores

##### Two Dim Data Plot #####

def get_two_dim_data_plot(*, traj_dict: dict, state_discriminator: LoosyDiscriminator):
    # get state discriminator hyperplane
    state_discr_params = state_discriminator.state.params
    b, n = jax.tree.flatten(state_discr_params)[0][-2:]
    assert n.shape[-1] == 1, n.ndim == 2
    b, n = b.squeeze(-1), n.squeeze(-1)

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

        a /= np.linalg.norm(a)

        return a, x0

    a, x0 = get_hyperplane_a_and_x0(n, b)
    assert (
        np.isclose((a * n).sum(), 0., atol=1e-4) and
        np.isclose((x0 * n).sum(), -b, atol=1e-4)
    ), f"{(a * n).sum() = } and {(x0 * n).sum() = },and {b = }"

    ## project mean of trajectories to the hyperplane
    traj_mean = np.concatenate([traj_dict["TR"], traj_dict["SE"]]).mean(0)
    traj_mean_proj = project_a_to_b(traj_mean - x0, a)
    x0 = x0 + traj_mean_proj

    ## two line's points
    h1 = x0 + a
    h2 = x0 - a

    # plot
    figsize=(5, 5)
    figure = plt.figure(figsize=figsize)

    plt.scatter(traj_dict["TR"][:, 0], traj_dict["TR"][:, 1], **TRAJECTORIES_SCATTER_PARAMS["TR"])
    plt.scatter(traj_dict["SE"][:, 0], traj_dict["SE"][:, 1], **TRAJECTORIES_SCATTER_PARAMS["SE"])
    plt.plot([h1[0], h2[0]], [h1[1], h2[1]], color="k")
    plt.close()

    return figure
