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
            x=state_pairs[k],
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
    source_state = domain_encoder.source_encoder
    loss_fn = domain_encoder.target_encoder.state.loss_fn

    # divergence score
    ## source expert
    _, source_expert_state_grad = jax.vmap(jax.value_and_grad(loss_fn.source_state_loss, has_aux=True))(
        source_state.params,
        state=source_state,
        discriminator=domain_encoder.state_discriminator,
        states=source_expert_batch["observations"],
    )
    _, source_expert_policy_grad = jax.vmap(jax.value_and_grad(loss_fn.source_policy_loss, has_aux=True))(
        source_state.params,
        state=source_state,
        discriminator=domain_encoder.state_discriminator,
        states=source_expert_batch["observations"],
    )
    source_expert_divergence_score = divergence_score_fn(state_grad=source_expert_state_grad, policy_grad=source_expert_policy_grad)

    return {"source_expert_diverence_score": source_expert_divergence_score}

def divergence_score_fn(state_grad: jnp.ndarray, policy_grad: jnp.ndarray):
    projection = project_a_to_b(a=policy_grad, b=state_grad)
    diverged_part = jnp.clip(projection, max=0.)
    diverged_part_norm = jnp.linalg.norm(diverged_part, axis=1).mean()
    state_grad_norm = jnp.linalg.norm(state_grad, axis=1).mean()
    return diverged_part_norm / state_grad_norm

def project_a_to_b(a: jnp.ndarray, b: jnp.ndarray):
    return cosine_similarity_fn(a, b) * a

def cosine_similarity_fn(a, b):
    return scalar_product_fn(a, b) / scalar_product_fn(a, a)**0.5 / scalar_product_fn(b, b)**0.5

def scalar_product_fn(a, b):
    return (a * b).sum(-1)
