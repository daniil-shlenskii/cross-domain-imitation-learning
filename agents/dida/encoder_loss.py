import jax
import jax.numpy as jnp

from gan.discriminator import Discriminator
from gan.losses import g_nonsaturating_loss
from nn.train_state import TrainState
from utils.types import Params, DataType


def encoder_loss_fn(
    params: Params,
    state: TrainState,
    batch: DataType,
    policy_discriminator: Discriminator,
    domain_discriminator: Discriminator,
    domain_loss_scale: float,
    is_learner_encoder: bool,
):
    batch["observations"] = state.apply_fn({"params": params}, batch["observations"], train=True)
    batch["observations_next"] = state.apply_fn({"params": params}, batch["observations_next"], train=True)

    policy_batch = jnp.concatenate([batch["observations"], batch["observations_next"]], axis=1)
    policy_logits = policy_discriminator(policy_batch)
    policy_loss = jax.lax.cond(
        is_learner_encoder,
        lambda policy_logits: g_nonsaturating_loss(policy_logits),
        lambda policy_logits: -g_nonsaturating_loss(policy_logits),
        policy_logits
    )

    domain_batch = batch["observations"]
    domain_logits = domain_discriminator(domain_batch)
    domain_loss = jax.lax.cond(
        is_learner_encoder,
        lambda domain_logits: -g_nonsaturating_loss(domain_logits),
        lambda domain_logits: g_nonsaturating_loss(domain_logits),
        domain_logits
    )

    loss = policy_loss + domain_loss_scale * domain_loss
    info = {
        f"{state.info_key}_loss": loss,
        f"{state.info_key}_policy_loss": policy_loss,
        f"{state.info_key}_domain_loss": domain_loss,
        "encoded_batch": batch,
    }
    return loss, info