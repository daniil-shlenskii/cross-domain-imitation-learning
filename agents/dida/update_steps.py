import functools

import jax
import jax.numpy as jnp
from agents.base_agent import Agent
from agents.gail.gail_discriminator import GAILDiscriminator
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils.types import Buffer, BufferState, DataType, PRNGKey


@jax.jit
def update_encoders_and_domain_discrimiantor(
    *,
    batch: DataType,
    expert_batch: DataType,
    learner_encoder: Generator,
    expert_encoder: Generator,
    policy_discriminator: GAILDiscriminator,
    domain_discriminator: Discriminator,
    domain_loss_scale: float,
):
    # update encoders
    new_learner_encoder, learner_encoder_info, learner_encoder_stats_info = learner_encoder.update(
        batch=batch,
        policy_discriminator=policy_discriminator,
        domain_discriminator=domain_discriminator,
        domain_loss_scale=domain_loss_scale,
        is_learner_encoder=True,
    )
    new_expert_encoder, expert_encoder_info, expert_encoder_stats_info = expert_encoder.update(
        batch=expert_batch,
        policy_discriminator=policy_discriminator,
        domain_discriminator=domain_discriminator,
        domain_loss_scale=domain_loss_scale,
        is_learner_encoder=False,
    )
    encoded_batch = learner_encoder_info.pop("encoded_batch")
    encoded_expert_batch = expert_encoder_info.pop("encoded_batch")

    # update domain discriminator
    new_domain_disc, domain_disc_info, domain_disc_stats_info = domain_discriminator.update(
        real_batch=encoded_batch["observations"],
        fake_batch=encoded_expert_batch["observations"],
        return_logits=True,
    )
    learner_domain_logits = domain_disc_info.pop("real_logits")
    expert_domain_logits = domain_disc_info.pop("fake_logits")

    info = {**learner_encoder_info, **expert_encoder_info, **domain_disc_info}
    stats_info = {**learner_encoder_stats_info, **expert_encoder_stats_info, **domain_disc_stats_info}
    return (
        new_learner_encoder,
        new_expert_encoder,
        new_domain_disc,
        encoded_batch,
        encoded_expert_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info,
    )

@jax.jit
def update_gail(
    *,
    batch: DataType,
    expert_batch: DataType,
    mixed_batch: DataType,
    #
    agent: Agent,
    policy_discriminator: GAILDiscriminator
):
    policy_batch = jnp.concatenate([batch["observations"], batch["observations_next"]], axis=1)
    expert_policy_batch = jnp.concatenate([expert_batch["observations"], expert_batch["observations_next"]], axis=1)
    mixed_policy_batch = jnp.concatenate([mixed_batch["observations"], mixed_batch["observations_next"]], axis=1)

    # update agent
    batch["reward"] = policy_discriminator.get_rewards(policy_batch)
    new_agent, agent_info, agent_stats_info = agent.update(batch)

    # update discriminator
    new_disc, disc_info, disc_stats_info = policy_discriminator.update(learner_batch=mixed_policy_batch, expert_batch=expert_policy_batch)

    info = {**agent_info, **disc_info}
    stats_info = {**agent_stats_info, **disc_stats_info}
    return new_agent, new_disc, info, stats_info


@functools.partial(jax.jit, static_argnames="expert_buffer")
def update_domain_discriminator_jit(
    *,
    rng: PRNGKey,
    batch: DataType,
    expert_buffer: Buffer,
    expert_buffer_state: BufferState,
    learner_encoder: Generator,
    expert_encoder: Generator,
    domain_discriminator: Discriminator
):
    new_rng, key = jax.random.split(rng, 2)
    expert_batch = expert_buffer.sample(expert_buffer_state, key).experience

    encoded_learner_domain_batch = learner_encoder(batch["observations"])
    encoded_expert_domain_batch = expert_encoder(expert_batch["observations"])

    new_domain_disc, domain_disc_info, domain_disc_stats_info = domain_discriminator.update(
        real_batch=encoded_learner_domain_batch,
        fake_batch=encoded_expert_domain_batch
    )
    return new_rng, new_domain_disc, domain_disc_info, domain_disc_stats_info