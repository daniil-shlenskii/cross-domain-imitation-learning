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
):
    # sample expert batch
    new_rng, sample_discr_expert_batch = sample_batch(
        dida_agent.rng, dida_agent.expert_buffer, dida_agent.expert_buffer_state
    )
    if dida_agent.sample_discriminator is not None:
        new_rng, expert_batch = dida_agent.sample_discriminator.sample(new_rng)
    else:
        expert_batch = sample_discr_expert_batch
    new_dida_agent = dida_agent.replace(rng=new_rng)

    # udpate encoders and domain discriminator
    domain_loss_scale = new_dida_agent.domain_loss_scale_fn(new_dida_agent)
    (
        new_dida_agent,
        batch,
        expert_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info,
    ) = new_dida_agent._update_encoders_and_domain_discrimiantor(
        batch=deepcopy(batch),
        expert_batch=expert_batch,
        domain_loss_scale=domain_loss_scale,
    )

    # encode sample_discr_expert_batch
    if dida_agent.sample_discriminator is not None:
        sample_discr_expert_batch["observations"] = dida_agent._preprocess_expert_observations(sample_discr_expert_batch["observations"])
        sample_discr_expert_batch["observations_next"] = dida_agent._preprocess_expert_observations(sample_discr_expert_batch["observations_next"])
    else:
        sample_discr_expert_batch = expert_batch

    # update dida_agent
    new_dida_agent = new_dida_agent.replace(rng=new_rng)
    info.update({"domain_loss_scale": domain_loss_scale})
    return (
        new_dida_agent,
        batch,
        expert_batch,
        sample_discr_expert_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info
    )
