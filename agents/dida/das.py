import jax
import jax.numpy as jnp

from gan.discriminator import Discriminator
from gan.generator import Generator
from utils.types import DataType, PRNGKey


def domain_adversarial_sampling(
    rng: PRNGKey,
    embedded_learner_batch: DataType,
    embedded_anchor_batch: DataType,
    domain_discriminator: Discriminator,
    alpha: float
):
    new_rng, perm_key, choice_key = jax.random.split(rng, 3)

    # preparation
    b_size = embedded_learner_batch["observations"].shape[0]
    num_to_mix = jnp.array(alpha * b_size, int)
    idcs_to_change = jax.random.permutation(perm_key, b_size).at[:num_to_mix].get()
    embedded_mixed_batch = embedded_anchor_batch

    # get das probabilites
    das_probs = get_das_probs(embedded_learner_batch, domain_discriminator)

    # udpate mixed batch with objects corresponding to the lowest das probabilites
    idcs = jax.random.choice(choice_key, a=b_size, shape=(num_to_mix,), p=1-das_probs)
    embedded_mixed_batch["observations"].at[idcs_to_change].set(embedded_learner_batch["observations"].at[idcs].get())
    embedded_mixed_batch["observations_next"].at[idcs_to_change].set(embedded_learner_batch["observations_next"].at[idcs].get())

    return new_rng, embedded_mixed_batch

def get_das_probs(
    embedded_learner_batch: DataType,
    domain_discriminator: Discriminator,
    clip_min: float = 0.1,
    clip_max: float = 0.9,
):
    observations_probs = jax.nn.sigmoid(domain_discriminator(embedded_learner_batch["observations"]))
    observations_next_probs = jax.nn.sigmoid(domain_discriminator(embedded_learner_batch["observations_next"]))
    probs = (observations_probs + observations_next_probs) * 0.5
    das_probs = probs / probs.sum()
    return jnp.clip(das_probs, min=clip_min, max=clip_max)