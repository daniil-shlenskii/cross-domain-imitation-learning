import jax
import jax.numpy as jnp

from agents.imitation_learning.utils import (
    get_random_from_expert_buffer_state, get_state_pairs, prepare_buffer)
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
        k: get_discriminator_score(
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
        k: get_discriminator_score(
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

def get_discriminator_score(discriminator: Discriminator, x: jnp.ndarray, is_real: bool):
    logits = discriminator(x)
    mask = jax.lax.cond(
        is_real,
        lambda logits: logits > 0.,
        lambda logits: logits < 0.,
        logits
    )
    score = mask.sum() / len(mask)
    return score

def get_discriminators_gradients_cosine_similarity(domain_encoder: "DomainEncoder", seed: int=0):
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

    # policy gradients
    policy_gradients = {
        k: jax.vmap(jax.grad(domain_encoder.policy_discriminator))(state_pair)
        for k, state_pair in state_pairs.items()
    }

    # state gradients
    state_gradients = {
        k: jax.vmap(jax.grad(domain_encoder.state_discriminator))(state_pair)
        for k, state_pair in state_pairs.items()
    }

    # cosine sim
    cosine_similarities = {
        k: (state_gradients[k] * policy_gradients[k]).sum(-1).mean(0) /
            jnp.linalg.norm(state_gradients[k]) * jnp.linalg.norm(policy_gradients[k])
        for k in batches
    }
    cosine_similarity = sum(cosine_similarities.values()) / len(cosine_similarities)

    return {
        "cosine_similarity": cosine_similarity,
        **{f"cosine_similarity_{k}": v for k, v in cosine_similarities.items()}
    }
