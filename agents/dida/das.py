import jax
import jax.numpy as jnp

from utils import sample_batch
from utils.types import DataType


@jax.jit
def _prepare_anchor_batch_jit(dida_agent: "DIDAAgent"):
    # sample anchor batch
    new_rng, anchor_batch = sample_batch(
        dida_agent.rng, dida_agent.expert_buffer, dida_agent.anchor_buffer_state
    )

    # encode anchor batch
    anchor_batch["observations"] = dida_agent.expert_encoder(anchor_batch["observations"])
    anchor_batch["observations_next"] = dida_agent.expert_encoder(anchor_batch["observations_next"])

    dida_agent = dida_agent.replace(rng=new_rng)
    return anchor_batch

class DomainAdversarialSampling:
    def __init__(
        self,
        sar_p: float,
        p_acc_ema: float,
        p_acc_ema_decay: float,
    ):
        self.sar_p = sar_p
        self.p_acc_ema = p_acc_ema
        self.p_acc_ema_decay = p_acc_ema_decay

    @property
    def key(self):
        if not hasattr(self, "rng"):
            self.rng = jax.random.key(0)
        self.rng, key = jax.random.split(self.rng)
        return key

    def mix_batches(
        self,
        learner_batch: DataType,
        anchor_batch: DataType,
        learner_domain_logits: jnp.ndarray,
        expert_domain_logits: jnp.ndarray,
    ):
        # SAR
        alpha, sar_info = self._get_self_adaptive_rate(learner_domain_logits, expert_domain_logits)

        # DAS confusion probs
        das_probs = self._get_das_probs(learner_domain_logits)

        # create mixed batch
        b_size = learner_batch["observations"].shape[0]
        num_to_mix = int(alpha * b_size)
        idcs = jax.random.choice(self.key, a=b_size, shape=(num_to_mix,), p=das_probs)

        mixed_batch = anchor_batch
        mixed_batch["observations"] = \
            mixed_batch["observations"].at[:num_to_mix].set(learner_batch["observations"].at[idcs].get())
        mixed_batch["observations_next"] = \
            mixed_batch["observations_next"].at[:num_to_mix].set(learner_batch["observations_next"].at[idcs].get())

        return mixed_batch, sar_info

    def _get_self_adaptive_rate(
        self,
        learner_domain_logits: jnp.ndarray,
        expert_domain_logits: jnp.ndarray,
    ):
        alpha, p_acc_ema, learner_score, expert_score, p_acc = _get_self_adaptive_rate_jit(
            learner_domain_logits=learner_domain_logits,
            expert_domain_logits=expert_domain_logits,
            p=self.sar_p,
            p_acc_ema=self.p_acc_ema,
            p_acc_ema_decay=self.p_acc_ema_decay,
        )
        sar_info = {
            "sar/alpha": alpha,
            "sar/p_acc_ema": p_acc_ema,
            "sar/learner_score": learner_score,
            "sar/expert_score": expert_score,
            "sar/p_acc": p_acc,
        }
        self.p_acc_ema = p_acc_ema
        return alpha, sar_info

    def _get_das_probs(self, learner_domain_logits: jnp.ndarray):
        return _get_das_probs_jit(learner_domain_logits)

@jax.jit
def _get_self_adaptive_rate_jit(
    learner_domain_logits: jnp.ndarray,
    expert_domain_logits: jnp.ndarray,
    p: float,
    p_acc_ema: float,
    p_acc_ema_decay: float,
):
    # compute accuracy of the domain discriminator
    learner_score = (learner_domain_logits < 0.).mean()
    expert_score = (expert_domain_logits > 0.).mean()

    p_acc = (learner_score + expert_score) * 0.5
    p_acc_ema = p_acc_ema * p_acc_ema_decay + p_acc * (1 - p_acc_ema_decay)

    # compute alpha
    alpha = jnp.minimum(p / p_acc_ema, (1 - p) / (1 - p_acc_ema))
    return alpha, p_acc_ema, learner_score, expert_score, p_acc


@jax.jit
def _get_das_probs_jit(learner_domain_logits: jnp.ndarray):
    shifted_logits = learner_domain_logits - learner_domain_logits.min()
    normalized_logits = shifted_logits / shifted_logits.max()
    confusion_probs = 1 - normalized_logits
    das_probs = confusion_probs / confusion_probs.sum()
    return das_probs
