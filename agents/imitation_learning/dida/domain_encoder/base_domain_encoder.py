from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from agents.imitation_learning.utils import (
    get_random_from_expert_buffer_state, get_state_pairs)
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils import SaveLoadFrozenDataclassMixin
from utils.types import Buffer, BufferState, DataType, PRNGKey
from utils.utils import sample_batch


class BaseDomainEncoder(PyTreeNode, SaveLoadFrozenDataclassMixin, ABC):
    rng: PRNGKey
    target_encoder: Generator
    state_discriminator: Discriminator
    policy_discriminator: Discriminator
    target_buffer: Buffer = struct.field(pytree_node=False)
    source_buffer: Buffer = struct.field(pytree_node=False)
    target_random_buffer_state: BufferState = struct.field(pytree_node=False)
    source_random_buffer_state: BufferState = struct.field(pytree_node=False)
    source_expert_buffer_state: BufferState = struct.field(pytree_node=False)
    state_loss_scale: float = struct.field(pytree_node=False)
    update_encoder_every: int = struct.field(pytree_node=False)
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        #
        target_buffer_config: DictConfig,
        target_random_buffer_state_config: DictConfig,
        #
        source_buffer: Buffer,
        source_expert_buffer_state: BufferState,
        #
        target_dim: int,
        encoding_dim: int ,
        #
        target_encoder_config: DictConfig,
        state_discriminator_config: DictConfig,
        policy_discriminator_config: DictConfig,
        #
        state_loss_scale: float,
        update_encoder_every: int = 1,
        #
        **kwargs,
    ):
        # target random buffer state init
        target_buffer = instantiate(target_buffer_config)
        target_random_buffer_state = instantiate(target_random_buffer_state_config)

        # source random buffer state init
        source_random_buffer_state = get_random_from_expert_buffer_state(
            seed=seed, expert_buffer_state=source_expert_buffer_state
        )

        # state discriminator init
        state_discriminator = instantiate(
            state_discriminator_config,
            seed=seed,
            input_dim=encoding_dim * 2,
            info_key="domain_encoder/state_discriminator",
            _recursive_=False,
        )

        # policy discriminator init
        policy_discriminator = instantiate(
            policy_discriminator_config,
            seed=seed,
            input_dim=encoding_dim * 2,
            info_key="domain_encoder/policy_discriminator",
            _recursive_=False,
        )

        # target encoder init
        target_encoder_config = OmegaConf.to_container(target_encoder_config)
        target_encoder_config["loss_config"]["state_loss"] = state_discriminator.state.loss_fn
        target_encoder_config["loss_config"]["policy_loss"] = policy_discriminator.state.loss_fn

        target_encoder = instantiate(
            target_encoder_config,
            seed=seed,
            input_dim=target_dim,
            output_dim=encoding_dim,
            info_key="target_encoder",
            _recursive_=False,
        )

        kwargs.pop("source_dim", None)
        return cls(
            rng=jax.random.key(seed),
            target_buffer=target_buffer,
            source_buffer=source_buffer,
            target_random_buffer_state=target_random_buffer_state,
            source_random_buffer_state=source_random_buffer_state,
            source_expert_buffer_state=source_expert_buffer_state,
            target_encoder=target_encoder,
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            state_loss_scale=state_loss_scale,
            update_encoder_every=update_encoder_every,
            _save_attrs=(
                "target_encoder",
                "state_discriminator",
                "policy_discriminator"
            ),
            **kwargs,
        )

    @jax.jit
    def encode_target_state(self, state: jnp.ndarray):
        return self.target_encoder(state)

    @abstractmethod
    def encode_source_state(self, state: jnp.ndarray):
        pass

    def update(self, target_expert_batch: DataType):
        (
            new_domain_encoder,
            target_random_batch,
            target_expert_batch,
            source_random_batch,
            source_expert_batch,
            info,
            stats_info,
        ) = _update_jit(
            domain_encoder=self,
            target_expert_batch=target_expert_batch
        )
        return (
            new_domain_encoder,
            target_random_batch,
            target_expert_batch,
            source_random_batch,
            source_expert_batch,
            info,
            stats_info,
        )

    def sample_batches(self):
        new_rng, target_random_batch = sample_batch(self.rng, self.target_buffer, self.target_random_buffer_state)
        new_rng, source_random_batch = sample_batch(new_rng, self.source_buffer, self.source_random_buffer_state)
        new_rng, source_expert_batch = sample_batch(new_rng, self.source_buffer, self.source_expert_buffer_state)

        return new_rng, target_random_batch, source_random_batch, source_expert_batch,

    @abstractmethod
    def _update_encoder(
        self,
        *,
        target_random_batch: DataType,
        target_expert_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
   ):
        pass

@jax.jit
def _update_jit(
    domain_encoder: BaseDomainEncoder,
    target_expert_batch: DataType,
):
    (
        new_rng,
        target_random_batch,
        source_random_batch,
        source_expert_batch,
    ) = domain_encoder.sample_batches()

    # update encoder
    new_domain_encoder, info, stats_info = domain_encoder._update_encoder(
        target_random_batch=target_random_batch,
        target_expert_batch=target_expert_batch,
        source_random_batch=source_random_batch,
        source_expert_batch=source_expert_batch,
    )
    target_random_batch = info.pop("target_random_batch")
    target_expert_batch = info.pop("target_expert_batch")
    source_random_batch = info.pop("source_random_batch")
    source_expert_batch = info.pop("source_expert_batch")

    new_domain_encoder = jax.lax.cond(
        (domain_encoder.state_discriminator.state.step + 1) % domain_encoder.update_encoder_every == 0,
        lambda: new_domain_encoder,
        lambda: domain_encoder,
    )

    # construct pairs
    target_random_pairs = get_state_pairs(target_random_batch)
    target_expert_pairs = get_state_pairs(target_expert_batch)
    source_random_pairs = get_state_pairs(source_random_batch)
    source_expert_pairs = get_state_pairs(source_expert_batch)

    # update state discriminator
    new_state_disc, state_disc_info, state_disc_stats_info = new_domain_encoder.state_discriminator.update(
        fake_batch=jnp.concatenate([target_random_pairs, target_expert_pairs]),
        real_batch=jnp.concatenate([source_random_pairs, source_expert_pairs]),
    )

    # update policy discriminator
    new_policy_disc, policy_disc_info, policy_disc_stats_info = new_domain_encoder.policy_discriminator.update(
        fake_batch=jnp.concatenate([target_random_pairs, source_random_pairs]),
        real_batch=jnp.concatenate([source_expert_pairs, source_expert_pairs]), # TODO: jax-based crutch for Gradient Penalty usage
    )

    # update domain encoder
    new_domain_encoder = new_domain_encoder.replace(
        rng=new_rng,
        state_discriminator=new_state_disc,
        policy_discriminator=new_policy_disc,
    )
    info.update({**state_disc_info, **policy_disc_info})
    stats_info.update({**state_disc_stats_info, **policy_disc_stats_info})

    return (
        new_domain_encoder,
        target_random_batch,
        target_expert_batch,
        source_random_batch,
        source_expert_batch,
        info,
        stats_info,
    )
