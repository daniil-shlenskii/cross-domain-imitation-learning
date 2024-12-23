import functools

import jax
import jax.numpy as jnp

from agents.imitation_learning.utils import get_state_pairs
from gan.discriminator import Discriminator


@jax.jit
def get_discriminators_scores(domain_encoder: "BaseDomainEncoder", seed: int=0):
    rng = jax.random.key(seed)

    # sample encoded batches
    rng, target_random_batch, source_random_batch, source_expert_batch = domain_encoder.sample_encoded_batches(rng)
    batches = {
        "target_random": target_random_batch,
        "source_random": source_random_batch,
        "source_expert": source_expert_batch,
    }

    # prepare state pairs
    state_pairs = {
        k: get_state_pairs(batch) for k, batch in batches.items()
    }

    # get policy scores
    is_reals = {
        "target_random": False,
        "source_random": False,
        "source_expert": True,
    }
    policy_scores = {
        k: _get_discriminator_score(
            discriminator=domain_encoder.policy_discriminator,
            x=state_pairs[k],
            is_real=is_reals[k],
        )
        for k in batches
    }
    policy_score = sum(policy_scores.values()) / len(policy_scores) # TODO: batches may have differnt size

    # get state scores
    is_reals = {
        "target_random": False,
        "source_random": True,
        "source_expert": True,
    }
    state_scores = {
        k: _get_discriminator_score(
            discriminator=domain_encoder.state_discriminator,
            x=batches[k]["observations"],
            is_real=is_reals[k],
        )
        for k in batches
    }
    state_score = sum(state_scores.values()) / len(state_scores)

    eval_info = {
        "policy_score": policy_score,
        "state_score": state_score,
        **{f"policy_{k}_score": score for k, score in policy_scores.items()},
        **{f"state_{k}_score": score for k, score in state_scores.items()},
    }
    return eval_info

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

def get_policy_discriminator_divergence_score(domain_encoder: "BaseDomainEncoder", seed: int=0):
    # sample batches
    rng = jax.random.key(seed)
    rng, target_random_batch, source_random_batch, source_expert_batch = domain_encoder.sample_batches(rng)

    #
    source_state = domain_encoder.source_encoder.state
    loss_fn = domain_encoder.target_encoder.state.loss_fn
    flatten_fn = lambda params_dict: jnp.concatenate([
        jnp.ravel(x) for x in
        jax.tree.flatten(params_dict, is_leaf=lambda x: isinstance(x, float))[0]
    ])

    # divergence score
    ## source expert
    ### state grad
    _, source_expert_state_grad = jax.value_and_grad(loss_fn.source_state_loss, has_aux=True)(
        source_state.params,
        state=source_state,
        discriminator=domain_encoder.state_discriminator,
        states=source_expert_batch["observations"],
    )
    source_expert_state_grad = flatten_fn(source_expert_state_grad)

    ### policy grad
    _, source_expert_policy_grad = jax.value_and_grad(loss_fn.source_policy_loss, has_aux=True)(
        source_state.params,
        state=source_state,
        discriminator=domain_encoder.policy_discriminator,
        states=source_expert_batch["observations"],
        states_next=source_expert_batch["observations_next"],
    )
    source_expert_policy_grad = flatten_fn(source_expert_policy_grad)

    se_neg_divergence_score, se_pos_divergence_score = divergence_scores_fn(
        state_grad=source_expert_state_grad,
        policy_grad=source_expert_policy_grad
    )

    return {
        "neg_divergence_score/source_expert": se_neg_divergence_score,
        "pos_divergence_score/source_expert": se_pos_divergence_score,
    }

def divergence_scores_fn(state_grad: jnp.ndarray, policy_grad: jnp.ndarray):
    projection = project_a_to_b(a=policy_grad, b=state_grad)

    neg_diverged_part = jnp.clip(projection, max=0.)
    neg_diverged_part_norm = jnp.linalg.norm(neg_diverged_part)

    pos_diverged_part = jnp.clip(projection, min=0.)
    pos_diverged_part_norm = jnp.linalg.norm(pos_diverged_part)

    state_grad_norm = jnp.linalg.norm(state_grad)

    neg_divergence_score = neg_diverged_part_norm / state_grad_norm
    pos_divergence_score = pos_diverged_part_norm / state_grad_norm

    return neg_divergence_score, pos_divergence_score

def project_a_to_b(a: jnp.ndarray, b: jnp.ndarray):
    return cosine_similarity_fn(a, b) * b

def cosine_similarity_fn(a, b):
    return scalar_product_fn(a, b) / scalar_product_fn(a, a)**0.5 / scalar_product_fn(b, b)**0.5

def scalar_product_fn(a, b):
    return (a * b).sum(-1)
