from typing import Any, Dict, Tuple

import jax.numpy as jnp
from nn.train_state import TrainState

from utils.custom_types import DataType, Params, PRNGKey


def actor_loss_fn(
    params: Params,
    state: TrainState,
    batch: DataType,
    critic1: TrainState,
    critic2: TrainState,
    temp: float,
    key: PRNGKey,
) -> Tuple[float, Dict[str, Any]]:
    dist = state.apply_fn({"params": params}, batch["observations"], train=True)
    actions, log_prob = dist.sample_and_log_prob(seed=key)

    state_action_value = jnp.minimum(
        critic1(batch["observations"], actions),
        critic2(batch["observations"], actions)
    )

    loss = (log_prob * temp - state_action_value).mean()
    return loss, {f"{state.info_key}_loss": loss, "entropy": -log_prob.mean()}

def critic_loss_fn(
    params: Params,
    state: TrainState,
    batch: DataType,
    target: jnp.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    preds = state.apply_fn({"params": params}, batch["observations"], batch["actions"], train=True)
    loss = ((preds - target)**2).mean()
    return loss, {f"{state.info_key}_loss": loss}

def temperature_loss_fn(
    params: Params,
    state: TrainState,
    entropy: float,
    target_entropy: float
) -> Tuple[float, Dict[str, Any]]:
    temp = state.apply_fn({"params": params})
    loss =  temp * (entropy - target_entropy).mean()
    return loss, {state.info_key: temp, f"{state.info_key}_loss": loss}
