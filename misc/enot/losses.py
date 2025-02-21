import jax.numpy as jnp

from nn.train_state import TrainState
from utils.custom_types import Params


def expectile_loss(diff: jnp.ndarray, expectile: float) -> jnp.ndarray:
    weight = jnp.where(diff >= 0, expectile, (1 - expectile))
    return weight * diff**2

def transport_loss(
    params: Params,
    state: TrainState,
    source: jnp.ndarray,
    enot,
):
    target_hat = state.apply_fn({"params": params}, source)
    loss = (enot.cost_fn(source, target_hat) - enot.g_potential(target_hat)).mean()
    return loss, {
        f"{state.info_key}/loss": loss,
        "target_hat": target_hat,
    }

def g_potential_loss(
    params: Params,
    state: TrainState,
    source: jnp.ndarray,
    target: jnp.ndarray,
    target_hat: jnp.ndarray,
    enot,
):
    g_values = state.apply_fn({"params": params}, target)
    g_hat_values = state.apply_fn({"params": params}, target_hat)
    downstream_loss = (g_hat_values - g_values * enot.target_weight).mean()
    reg_loss = expectile_loss(
        diff=(
            enot.cost_fn(source, target_hat) -
            enot.cost_fn(source, target) +
            g_values * enot.target_weight -
            g_hat_values
        ),
        expectile=enot.expectile,
    ).mean()
    loss = downstream_loss + reg_loss * enot.expectile_loss_coef
    return loss, {
        f"{state.info_key}/loss": loss,
        f"{state.info_key}/downstream_loss": downstream_loss,
        f"{state.info_key}/reg_loss": reg_loss,
    }
