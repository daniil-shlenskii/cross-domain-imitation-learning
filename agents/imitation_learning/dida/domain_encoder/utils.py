import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from agents.imitation_learning.utils import get_state_pairs
from gan.discriminator import Discriminator


def get_states_tsne_scatterplots(
    domain_encoder: "BaseDomainEncoder",
    seed: int,
):
    observation_keys = ("observations", "observations_next")

    # get trajectories
    target_random_trajs = domain_encoder.target_random_buffer_state.experience
    source_random_trajs = domain_encoder.source_random_buffer_state.experience
    source_expert_trajs = domain_encoder.source_expert_buffer_state.experience

    end_of_firt_traj_idx = np.argmax(target_random_trajs["truncated"][0])
    target_random_traj = {k: target_random_trajs[k][0, :end_of_firt_traj_idx] for k in observation_keys}
    source_random_traj = {k: source_random_trajs[k][0, :end_of_firt_traj_idx] for k in observation_keys}
    source_expert_traj = {k: source_expert_trajs[k][0, :end_of_firt_traj_idx] for k in observation_keys}

    # encode trjectories
    target_random_traj = domain_encoder.encode_target_batch(target_random_traj)
    source_random_traj = domain_encoder.encode_source_batch(source_random_traj)
    source_expert_traj = domain_encoder.encode_source_batch(source_expert_traj)

    # get state pairs (in order to anchor to be different from expert buffer)
    target_random_traj = get_state_pairs(target_random_traj)
    source_random_traj = get_state_pairs(source_random_traj)
    source_expert_traj = get_state_pairs(source_expert_traj)

    # combine all states for further processing
    states = [target_random_traj, source_random_traj, source_expert_traj] 
    states_sizes_list = [0] + list(np.cumsum(list(map(
        lambda embs: embs.shape[0],
        states,
    ))))

    # tsne
    tsne_states_united= TSNE(random_state=seed).fit_transform(np.concatenate(states))
    tsne_states = [
        tsne_states_united[
            states_sizes_list[i]: states_sizes_list[i + 1]
        ]
        for i in range(len(states))
    ]

    # scatterplots
    opaque = np.linspace(.2, 1., num=end_of_firt_traj_idx)
    scatter_params_list = (
        {"label": "TR", "c": "tab:green",  "marker": "x", "alpha": opaque},
        {"label": "SR", "c": "tab:orange", "marker": "s", "alpha": opaque},
        {"label": "SE", "c": "tab:red",    "marker": "o", "alpha": opaque},
    )
    figsize=(5, 5)

    states_figure = plt.figure(figsize=figsize)
    for tsne_state_embeddings, scatter_params in zip(tsne_states, scatter_params_list):
        plt.scatter(tsne_state_embeddings[:, 0], tsne_state_embeddings[:, 1], **scatter_params)
    plt.legend()
    plt.close()

    return states_figure

def get_discriminators_scores(domain_encoder: "BaseDomainEncoder", seed: int=0):
    rng = jax.random.key(seed)

    # sample encoded batches
    rng, target_random_batch, source_random_batch, source_expert_batch = domain_encoder.sample_encoded_batches(rng)
    batches = {
        "target_random": target_random_batch,
        "source_random": source_random_batch,
        "source_expert": source_expert_batch,
    }

    # prepare states and state pairs
    states = {
        k: batch["observations"] for k, batch in batches.items()
    }
    state_pairs = {
        k: get_state_pairs(batch) for k, batch in batches.items()
    }

    # get scores
    def _get_scores(discriminator, inputs, is_reals):
        scores = {
            k: _get_discriminator_score(
                discriminator=discriminator,
                x=inputs[k],
                is_real=is_reals[k],
            )
            for k in batches
        }
        return scores

    ## get state scores
    is_reals = {
        "target_random": False,
        "source_random": True,
        "source_expert": True,
    }
    if domain_encoder.discriminators.has_state_discriminator_paired_input:
        state_scores = _get_scores(domain_encoder.state_discriminator, state_pairs, is_reals)
    else:
        state_scores = _get_scores(domain_encoder.state_discriminator, states, is_reals)
    state_score = sum(state_scores.values()) / len(state_scores)

    # get policy scores
    is_reals = {
        "target_random": False,
        "source_random": False,
        "source_expert": True,
    }
    policy_scores = _get_scores(
        domain_encoder.policy_discriminator,
        state_pairs,
        is_reals,
    )
    policy_score = sum(policy_scores.values()) / len(policy_scores) # TODO: batches may have differnt size

    eval_info = {
            "policy_score": policy_score,
            "state_score": state_score,
            **{f"policy_{k}_score": score for k, score in policy_scores.items()},
            **{f"state_{k}_score": score for k, score in state_scores.items()},
        }
    return eval_info

@jax.jit
def _get_discriminator_score(discriminator: Discriminator, x: jnp.ndarray, is_real: bool):
    logits = discriminator(x)
    mask = jax.lax.cond(
        is_real,
        lambda logits: logits > 0.,
        lambda logits: logits < 0.,
        logits
    )
    score = mask.sum() / len(mask)
    return score

def get_policy_discriminator_divergence_score_params(domain_encoder: "BaseDomainEncoder", seed: int=0):
    # sample batches
    rng = jax.random.key(seed)
    rng, target_random_batch, source_random_batch, source_expert_batch = domain_encoder.sample_batches(rng)

    #
    source_state = domain_encoder.source_encoder.state
    loss_fn = domain_encoder.target_encoder.state.loss_fn
    flatten_fn = lambda params_dict: jnp.concatenate([
        jnp.ravel(x) for x in
        jax.tree.flatten(params_dict, is_leaf=lambda x: isinstance(x, jnp.ndarray))[0]
    ])

    # divergence score
    ## source expert
    ### state grad
    _, source_expert_state_grad = jax.value_and_grad(loss_fn.source_state_loss, has_aux=True)(
        source_state.params,
        state=source_state,
        discriminator=domain_encoder.state_discriminator,
        states=source_expert_batch["observations"],
        states_next=source_expert_batch["observations_next"],
     )
    source_expert_state_grad = flatten_fn(source_expert_state_grad)

    ### policy grad
    _, source_expert_policy_grad = jax.value_and_grad(loss_fn.source_expert_policy_loss, has_aux=True)(
        source_state.params,
        state=source_state,
        discriminator=domain_encoder.policy_discriminator,
        states=source_expert_batch["observations"],
        states_next=source_expert_batch["observations_next"],
    )
    source_expert_policy_grad = flatten_fn(source_expert_policy_grad)

    se_divergence_score = divergence_scores_fn(
        state_grad=source_expert_state_grad,
        policy_grad=source_expert_policy_grad
    )

    return {"divergence_score_params/source_expert": se_divergence_score,}

def get_policy_discriminator_divergence_score_embeddings(domain_encoder: "BaseDomainEncoder", seed: int=0):
    # sample batches
    rng = jax.random.key(seed)
    rng, target_random_batch, source_random_batch, source_expert_batch = domain_encoder.sample_encoded_batches(rng)

    # get_pairs
    source_expert_pairs = get_state_pairs(source_expert_batch)

    #
    loss_fn = domain_encoder.target_encoder.state.loss_fn

    # divergence score
    ## source expert
    ### state grad
    state_discriminator_input = source_expert_batch["observations"]
    if domain_encoder.discriminators.has_state_discriminator_paired_input:
        state_discriminator_input = source_expert_pairs
    source_expert_state_grad = jax.grad(
        lambda x: loss_fn.real_state_loss_fn(domain_encoder.state_discriminator(x))
    )(state_discriminator_input)
    source_expert_state_grad = source_expert_state_grad.mean(0)

    ### policy grad
    source_expert_policy_grad = jax.grad(
        lambda x: loss_fn.real_policy_loss_fn(domain_encoder.policy_discriminator(x))
    )(source_expert_pairs)
    source_expert_policy_grad = source_expert_policy_grad.mean(0)
    source_expert_policy_grad = source_expert_policy_grad.at[:source_expert_state_grad.shape[-1]].get()

    se_divergence_score = divergence_scores_fn(
        state_grad=source_expert_state_grad,
        policy_grad=source_expert_policy_grad
    )

    return {"divergence_score_embeddings/source_expert": se_divergence_score,}

def divergence_scores_fn(state_grad: jnp.ndarray, policy_grad: jnp.ndarray):
    projection = project_a_to_b(a=policy_grad, b=state_grad)

    projection_norm = jnp.linalg.norm(projection)
    state_grad_norm = jnp.linalg.norm(state_grad)

    if projection_norm == 0.:
        s = 1.
    else:
        s = jnp.sign(cosine_similarity_fn(state_grad, projection))

    divergence_score = s * projection_norm / state_grad_norm

    return divergence_score

def project_a_to_b(a: jnp.ndarray, b: jnp.ndarray):
    return cosine_similarity_fn(a, b) * b

def cosine_similarity_fn(a: jnp.ndarray, b: jnp.ndarray):
    return scalar_product_fn(a, b) / scalar_product_fn(a, a)**0.5 / scalar_product_fn(b, b)**0.5

def scalar_product_fn(a: jnp.ndarray, b: jnp.ndarray):
    return (a * b).sum()
