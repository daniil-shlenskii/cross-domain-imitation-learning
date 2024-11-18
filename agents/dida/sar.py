import jax
import jax.numpy as jnp
from gan.discriminator import Discriminator
from utils.types import DataType


def self_adaptive_rate(
    learner_domain_logits: float,
    expert_domain_logits: float,
    p: float,
    #
    p_acc_ema: float,
    p_acc_ema_decay: float,
):
    # compute accuracy of the domain discriminator
    learner_probs = jax.nn.sigmoid(learner_domain_logits)
    expert_probs = jax.nn.sigmoid(expert_domain_logits)

    learner_score = (learner_probs > 0.5).mean()
    expert_score = (expert_probs < 0.5).mean()

    p_acc = (learner_score + expert_score) * 0.5
    p_acc_ema = p_acc_ema * p_acc_ema_decay + p_acc * (1 - p_acc_ema_decay)

    # compute alpha
    alpha = jnp.minimum(p_acc_ema / p, (1 - p_acc_ema) / (1 - p))

    # info = {"sar/alpha": alpha, "sar/p_acc": p_acc, "sar/p_acc_ema": p_acc_ema}
    return alpha, p_acc_ema, p_acc
