from copy import deepcopy

import jax.numpy as jnp

from agents.imitation_learning.cross_domain.domain_encoder.utils import \
    encode_states_given_params
from nn.train_state import TrainState
from utils.custom_types import DataType, Params


def expectile_loss(diff: jnp.ndarray, expectile: float) -> jnp.ndarray:
    weight = jnp.where(diff >= 0, expectile, (1 - expectile))
    return weight * diff**2

def transport_loss(
    params: Params,
    state: TrainState,
    source: DataType,
    enot,
):
    target_hat = deepcopy(source)
    target_hat["observations"], target_hat["observations_next"] =\
        encode_states_given_params(params, state, source["observations"], source["observations_next"])

    loss = (enot.cost_fn(source, target_hat) - enot.g_potential(target_hat["observations"])).mean()
    return loss, {
        f"{state.info_key}/loss": loss,
        "target_hat": target_hat,
    }

def g_potential_loss(
    params: Params,
    state: TrainState,
    source: DataType,
    target: DataType,
    target_hat: DataType,
    enot,
):
    g_values = state.apply_fn({"params": params}, target["observations"])
    g_hat_values = state.apply_fn({"params": params}, target_hat["observations"])
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
