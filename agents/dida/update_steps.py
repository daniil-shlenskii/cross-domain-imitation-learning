from copy import deepcopy

import jax

from utils import sample_batch
from utils.types import DataType


@jax.jit
def _update_domain_discriminator_only_jit(
    dida_agent: "DIDAAgent",
    batch: DataType,
):
    # sample expert batch
    new_rng, expert_batch = sample_batch(
        dida_agent.rng, dida_agent.expert_buffer, dida_agent.expert_buffer_state
    )

    # encode observations
    batch["observations"] = dida_agent.learner_encoder(batch["observations"])
    expert_batch["observations"] = dida_agent.expert_encoder(expert_batch["observations"])

    # update domain discriminator
    new_domain_disc, domain_disc_info, domain_disc_stats_info = dida_agent.domain_discriminator.update(
        real_batch=expert_batch["observations"],
        fake_batch=batch["observations"],
    )

    # update dida agent
    new_dida_agent = dida_agent.replace(
        rng=new_rng,
        domain_discriminator=new_domain_disc
    )
    return new_dida_agent, domain_disc_info, domain_disc_stats_info

@jax.jit
def _update_encoders_and_domain_discriminator_jit(
    dida_agent: "DIDAAgent",
    batch: DataType,
    domain_loss_scale: float,
):
    # sample expert batch
    new_rng, expert_batch = sample_batch(
        dida_agent.rng, dida_agent.expert_buffer, dida_agent.expert_buffer_state
    )

    # udpate encoders and domain discriminator
    (
        new_dida_agent,
        batch,
        expert_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info,
    ) = dida_agent._update_encoders_and_domain_discrimiantor(
        batch=deepcopy(batch),
        expert_batch=expert_batch,
        domain_loss_scale=domain_loss_scale,
    )

    # update dida_agent
    new_dida_agent = new_dida_agent.replace(rng=new_rng)

    return (
        new_dida_agent,
        batch,
        expert_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info
    )
