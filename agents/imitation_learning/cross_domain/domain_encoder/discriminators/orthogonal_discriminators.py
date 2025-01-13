import jax
import jax.numpy as jnp

from agents.imitation_learning.utils import get_state_pairs
from utils.custom_types import DataType

from .base_discriminators import BaseDomainEncoderDiscriminators


class OrthogonalDomainEncoderDiscriminators(BaseDomainEncoderDiscriminators):
    def update(
        self,
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        return _update_jit(
            discriminators=self,
            target_random_batch=target_random_batch,
            source_random_batch=source_random_batch,
            source_expert_batch=source_expert_batch,
        )

@jax.jit
def _update_jit(
    discriminators: OrthogonalDomainEncoderDiscriminators,
    target_random_batch: DataType,
    source_random_batch: DataType,
    source_expert_batch: DataType,
):
    # update policy discriminator
    ## construct pairs
    target_random_state_pairs = get_state_pairs(target_random_batch)
    source_random_state_pairs = get_state_pairs(source_random_batch)
    source_expert_state_pairs = get_state_pairs(source_expert_batch)

    ## update
    new_policy_disc, policy_disc_info, policy_disc_stats_info = discriminators.policy_discriminator.update(
        target_random_state_pairs=target_random_state_pairs,
        source_random_state_pairs=source_random_state_pairs,
        source_expert_state_pairs=source_expert_state_pairs,
    )
    new_policy_disc = jax.lax.cond(
        (discriminators.state_discriminator.state.step + 1) % discriminators.update_policy_discriminator_every == 0,
        lambda: new_policy_disc,
        lambda: discriminators.policy_discriminator,
    )
    target_random_state_pairs_grad = policy_disc_info.pop("target_random_state_pairs_grad")
    source_expert_state_pairs_grad = policy_disc_info.pop("source_expert_state_pairs_grad")

    # update state discriminator
    ## prepare grads
    dim = target_random_batch["observations"].shape[-1]
    target_random_states_policy_grad = target_random_state_pairs_grad.at[:, :dim].get()
    source_expert_states_policy_grad = source_expert_state_pairs_grad.at[:, :dim].get()

    ## update
    new_state_disc, state_disc_info, state_disc_stats_info = discriminators.state_discriminator.update(
        target_random_states=target_random_batch["observations"],
        source_expert_states=source_expert_batch["observations"],
        target_random_states_policy_grad=target_random_states_policy_grad,
        source_expert_states_policy_grad=source_expert_states_policy_grad,
    )

    # final update
    new_discriminators = discriminators.replace(
        state_discriminator=new_state_disc,
        policy_discriminator=new_policy_disc,
    )
    info = {**state_disc_info, **policy_disc_info}
    stats_info = {**state_disc_stats_info, **policy_disc_stats_info}

    return new_discriminators, info, stats_info
