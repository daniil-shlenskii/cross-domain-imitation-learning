import jax
import jax.numpy as jnp

from gan.base_losses import (g_nonsaturating_logistic_loss,
                             g_nonsaturating_softplus_loss)
from gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.types import DataType, Params


def learner_encoder_loss(
    params: Params,
    state: TrainState,
    batch: DataType,
    policy_discriminator: Discriminator,
    domain_discriminator: Discriminator,
    domain_loss_scale: float,
):
    batch["observations"] = state.apply_fn({"params": params}, batch["observations"], train=True)
    batch["observations_next"] = state.apply_fn({"params": params}, batch["observations_next"], train=True)

    policy_batch = jnp.concatenate([batch["observations"], batch["observations_next"]], axis=1)
    policy_logits = policy_discriminator(policy_batch)
    policy_loss = g_nonsaturating_softplus_loss(policy_logits)

    domain_batch = batch["observations"]
    domain_logits = domain_discriminator(domain_batch)
    domain_loss = -g_nonsaturating_logistic_loss(domain_logits)

    loss = policy_loss + domain_loss_scale * domain_loss
    info = {
        f"{state.info_key}_loss": loss,
        f"{state.info_key}_policy_loss": policy_loss,
        f"{state.info_key}_domain_loss": domain_loss,
        "encoded_batch": batch,
    }
    return loss, info

def expert_encoder_loss(
    params: Params,
    state: TrainState,
    batch: DataType,
    policy_discriminator: Discriminator,
    domain_discriminator: Discriminator,
    domain_loss_scale: float,
):
    batch["observations"] = state.apply_fn({"params": params}, batch["observations"], train=True)
    batch["observations_next"] = state.apply_fn({"params": params}, batch["observations_next"], train=True)

    policy_batch = jnp.concatenate([batch["observations"], batch["observations_next"]], axis=1)
    policy_logits = policy_discriminator(policy_batch)
    policy_loss = g_nonsaturating_softplus_loss(-policy_logits)

    domain_batch = batch["observations"]
    domain_logits = domain_discriminator(domain_batch)
    domain_loss = g_nonsaturating_logistic_loss(domain_logits)

    loss = policy_loss + domain_loss_scale * domain_loss
    info = {
        f"{state.info_key}_loss": loss,
        f"{state.info_key}_policy_loss": policy_loss,
        f"{state.info_key}_domain_loss": domain_loss,
        "encoded_batch": batch,
    }
    return loss, info
