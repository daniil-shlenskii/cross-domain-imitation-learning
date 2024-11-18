import jax
import jax.numpy as jnp
import numpy as np
from agents.dida.sar import self_adaptive_rate
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils.types import DataType, PRNGKey


def domain_adversarial_sampling(
    rng: PRNGKey,
    encoded_learner_batch: DataType,
    encoded_anchor_batch: DataType,
    learner_domain_logits: jnp.ndarray,
    #
    expert_domain_logits: jnp.ndarray,
    sar_p: float,
    p_acc_ema: float,
    p_acc_ema_decay: float,
):
    # compute sar and das probs
    (
        das_probs,
        alpha,
        new_p_acc_ema,
        p_acc,
    ) = _compute_sar_and_das_probs_jit(
        learner_domain_logits=learner_domain_logits,
        expert_domain_logits=expert_domain_logits,
        sar_p=sar_p,
        p_acc_ema=p_acc_ema,
        p_acc_ema_decay=p_acc_ema_decay,
    )

    # mixed batch creation
    new_rng, key = jax.random.split(rng)

    b_size = encoded_learner_batch["observations"].shape[0]
    num_to_mix = int(alpha * b_size)
    idcs = jax.random.choice(key, a=b_size, shape=(num_to_mix,), p=das_probs)

    encoded_mixed_batch = encoded_anchor_batch
    encoded_mixed_batch["observations"] = \
        encoded_mixed_batch["observations"].at[:num_to_mix].set(encoded_learner_batch["observations"].at[idcs].get())
    encoded_mixed_batch["observations_next"] = \
        encoded_mixed_batch["observations_next"].at[:num_to_mix].set(encoded_learner_batch["observations_next"].at[idcs].get())

    sar_info = {"sar/alpha": alpha, "sar/p_acc": p_acc, "sar/p_acc_ema": p_acc_ema}
    return new_rng, encoded_mixed_batch, new_p_acc_ema, sar_info

@jax.jit
def _compute_sar_and_das_probs_jit(
    *,
    learner_domain_logits: jnp.ndarray,
    expert_domain_logits: jnp.ndarray,
    sar_p: float,
    p_acc_ema: float,
    p_acc_ema_decay: float,
):
    # compute sar
    alpha, new_p_acc_ema, p_acc = self_adaptive_rate(
        learner_domain_logits=learner_domain_logits,
        expert_domain_logits=expert_domain_logits,
        p=sar_p,
        p_acc_ema=p_acc_ema,
        p_acc_ema_decay=p_acc_ema_decay,
    )

    # compute das probs
    probs = jax.nn.sigmoid(learner_domain_logits)
    confusion_probs = 1 - probs
    das_probs = confusion_probs / confusion_probs.sum()

    return (
        das_probs,
        alpha,
        new_p_acc_ema,
        p_acc,
    )
