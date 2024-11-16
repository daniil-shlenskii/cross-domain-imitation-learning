import jax
import jax.numpy as jnp

from gan.discriminator import Discriminator
from utils.types import DataType


@jax.jit
def self_adaptive_rate(
    domain_discriminator: Discriminator,
    learner_batch: DataType,
    expert_batch: DataType,
    p: float,
    #
    p_acc_ema: float,
    p_acc_ema_decay: float,
):
    p_acc = _discriminator_accuracy_score_jit(domain_discriminator, learner_batch, expert_batch)
    p_acc_ema = p_acc_ema * p_acc_ema_decay + p_acc * (1 - p_acc_ema_decay)
    alpha = jnp.minimum(p_acc_ema / p, (1 - p_acc_ema) / (1 - p))
    info = {"sar/alpha": alpha, "sar/p_acc": p_acc, "sar/p_acc_ema": p_acc_ema}
    return alpha, p_acc_ema, info

def _discriminator_accuracy_score_jit(domain_discriminator, learner_batch, expert_batch):
    learner_probs = jax.nn.sigmoid(domain_discriminator(learner_batch["observations"]))
    expert_probs = jax.nn.sigmoid(domain_discriminator(expert_batch["observations"]))

    learner_score = (learner_probs > 0.5).mean()
    expert_score = (expert_probs < 0.5).mean()

    return (learner_score + expert_score) * 0.5