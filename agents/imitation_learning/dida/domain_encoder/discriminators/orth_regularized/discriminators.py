import jax
import jax.numpy as jnp

from agents.imitation_learning.dida.domain_encoder.discriminators.default.discriminators import \
    DomainEncoderDiscriminators
from agents.imitation_learning.utils import get_state_pairs
from utils.types import DataType


class OrthRegualirizedDomainEncoderDiscrimiantors(DomainEncoderDiscriminators):
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
    discriminators: OrthRegualirizedDomainEncoderDiscrimiantors,
    target_random_batch: DataType,
    source_random_batch: DataType,
    source_expert_batch: DataType,
):
    # update policy discriminator
    ## construct pairs
    target_random_pairs = get_state_pairs(target_random_batch)
    source_random_pairs = get_state_pairs(source_random_batch)
    source_expert_pairs = get_state_pairs(source_expert_batch)

    ## update
    new_policy_disc, policy_disc_info, policy_disc_stats_info = discriminators.policy_discriminator.update(
        fake_batch=jnp.concatenate([target_random_pairs, source_random_pairs]),
        real_batch=source_expert_pairs,
    )
    new_policy_disc = jax.lax.cond(
        (discriminators.state_discriminator.state.step + 1) % discriminators.update_policy_discriminator_every == 0,
        lambda: new_policy_disc,
        lambda: discriminators.policy_discriminator,
    )
    fake_pairs_grads = policy_disc_info.pop("fake_grads")
    real_pairs_grads = policy_disc_info.pop("real_grads")

    # update state discriminator
    ## prepare grads
    dim = target_random_batch["observations"].shape[-1]
    target_random_policy_grad = fake_pairs_grads.at[:target_random_pairs.shape[0], :dim].get()
    source_expert_policy_grad = real_pairs_grads.at[:, :dim].get()

    ## update
    new_state_disc, state_disc_info, state_disc_stats_info = discriminators.state_discriminator.update(
        fake_batch=target_random_batch["observations"],
        real_batch=source_expert_batch["observations"],
        fake_policy_grad=target_random_policy_grad,
        real_policy_grad=source_expert_policy_grad,
    )

    # final update
    new_discriminators = discriminators.replace(
        state_discriminator=new_state_disc,
        policy_discriminator=new_policy_disc,
    )
    info = {**state_disc_info, **policy_disc_info}
    stats_info = {**state_disc_stats_info, **policy_disc_stats_info}

    return new_discriminators, info, stats_info
