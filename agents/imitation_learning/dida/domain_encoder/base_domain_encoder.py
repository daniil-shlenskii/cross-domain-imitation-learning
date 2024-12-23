from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from agents.imitation_learning.dida.domain_encoder.utils import (
    get_discriminators_scores, get_policy_discriminator_divergence_score)
from agents.imitation_learning.utils import (
    get_random_from_expert_buffer_state, get_state_pairs, prepare_buffer)
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils import SaveLoadFrozenDataclassMixin
from utils.types import BufferState, DataType, PRNGKey
from utils.utils import sample_batch


class BaseDomainEncoder(PyTreeNode, SaveLoadFrozenDataclassMixin, ABC):
    rng: PRNGKey
    target_encoder: Generator
    state_discriminator: Discriminator
    policy_discriminator: Discriminator
    target_buffer: Any = struct.field(pytree_node=False)
    source_buffer: Any = struct.field(pytree_node=False)
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
        encoding_dim: int,
        #
        target_batch_size: int,
        target_random_buffer_state_path,
        #
        source_batch_size: int,
        source_expert_buffer_state_path,
        #
        target_encoder_config: DictConfig,
        state_discriminator_config: DictConfig,
        policy_discriminator_config: DictConfig,
        #
        state_loss_scale: float,
        update_encoder_every: int = 1,
        #
        source_buffer_state_processor_config: DictConfig = None,
        #
        **kwargs,
    ):
        # target random buffer state init
        target_buffer, target_random_buffer_state = prepare_buffer(
            buffer_state_path=target_random_buffer_state_path,
            batch_size=target_batch_size,
        )

        # source expert buffer state init
        source_buffer, source_expert_buffer_state = prepare_buffer(
            buffer_state_path=source_expert_buffer_state_path,
            batch_size=source_batch_size,
            buffer_state_processor_config=source_buffer_state_processor_config,
        )

        # source random buffer state init
        source_random_buffer_state = get_random_from_expert_buffer_state(
            seed=seed, expert_buffer_state=source_expert_buffer_state
        )

        # state discriminator init
        state_discriminator = instantiate(
            state_discriminator_config,
            seed=seed,
            input_dim=encoding_dim,
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

        target_dim = target_random_buffer_state.experience["observations"].shape[-1]

        target_encoder = instantiate(
            target_encoder_config,
            seed=seed,
            input_dim=target_dim,
            output_dim=encoding_dim,
            info_key="target_encoder",
            _recursive_=False,
        )

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

    def __getattr__(self, item) -> Any:
        if item == "source_encoder":
            return self.target_encoder
        return super().__getattribute__(item)

    @jax.jit
    def encode_target_state(self, state: jnp.ndarray):
        return self.target_encoder(state)

    @jax.jit
    def encode_source_state(self, state: jnp.ndarray):
        return self.source_encoder(state)

    def pretrain_update(self):
        (
            new_domain_encoder,
            _,
            _,
            _,
            _,
            info,
            stats_info,
        ) = _pretrain_update_jit(domain_encoder=self)
        return new_domain_encoder, info, stats_info

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

    def evaluate(self, seed: int=0):
        scores = get_discriminators_scores(domain_encoder=self, seed=seed)
        divergence_scores = get_policy_discriminator_divergence_score(domain_encoder=self, seed=seed)
        eval_info = {**scores, **divergence_scores}
        return eval_info

    def sample_batches(self, rng: PRNGKey):
        new_rng, target_random_batch = sample_batch(rng, self.target_buffer, self.target_random_buffer_state)
        new_rng, source_random_batch = sample_batch(new_rng, self.source_buffer, self.source_random_buffer_state)
        new_rng, source_expert_batch = sample_batch(new_rng, self.source_buffer, self.source_expert_buffer_state)
        return new_rng, target_random_batch, source_random_batch, source_expert_batch,

    def encode_target_batch(self, batch):
        batch = deepcopy(batch)
        batch["observations"] = self.encode_target_state(batch["observations"])
        batch["observations_next"] = self.encode_target_state(batch["observations_next"])
        return batch

    def encode_source_batch(self, batch):
        batch = deepcopy(batch)
        batch["observations"] = self.encode_source_state(batch["observations"])
        batch["observations_next"] = self.encode_source_state(batch["observations_next"])
        return batch

    def sample_encoded_batches(self, rng: PRNGKey):
        new_rng, target_random_batch, source_random_batch, source_expert_batch = self.sample_batches(rng)

        for k in ["observations", "observations_next"]:
            target_random_batch[k] = self.encode_target_state(target_random_batch[k])
            source_random_batch[k] = self.encode_source_state(source_random_batch[k])
            source_expert_batch[k] = self.encode_source_state(source_expert_batch[k])
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
def _pretrain_update_jit(domain_encoder: BaseDomainEncoder):
    (
        new_rng,
        target_random_batch,
        source_random_batch,
        source_expert_batch,
    ) = domain_encoder.sample_batches(domain_encoder.rng)
    new_domain_encoder = domain_encoder.replace(rng=new_rng)
    return _update(
        domain_encoder=new_domain_encoder,
        target_random_batch=target_random_batch,
        target_expert_batch=target_random_batch,
        source_random_batch=source_random_batch,
        source_expert_batch=source_expert_batch,
    )

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
    ) = domain_encoder.sample_batches(domain_encoder.rng)
    new_domain_encoder = domain_encoder.replace(rng=new_rng)
    return _update(
        domain_encoder=new_domain_encoder,
        target_random_batch=target_random_batch,
        target_expert_batch=target_expert_batch,
        source_random_batch=source_random_batch,
        source_expert_batch=source_expert_batch,
    )

def _update(
    domain_encoder: BaseDomainEncoder,
    target_random_batch: DataType,
    target_expert_batch: DataType,
    source_random_batch: DataType,
    source_expert_batch: DataType,
):
    # update encoder
    new_domain_encoder, info, stats_info = domain_encoder._update_encoder(
        target_random_batch=target_random_batch,
        target_expert_batch=target_expert_batch,
        source_random_batch=source_random_batch,
        source_expert_batch=source_expert_batch,
    )
    target_random_batch = info.pop("target_random_batch")
    source_expert_batch = info.pop("source_expert_batch")

    new_domain_encoder = jax.lax.cond(
        (domain_encoder.state_discriminator.state.step + 1) % domain_encoder.update_encoder_every == 0,
        lambda: new_domain_encoder,
        lambda: domain_encoder,
    )

    # encode target expert and source random batches
    target_expert_batch = domain_encoder.encode_target_batch(target_expert_batch)
    source_random_batch = domain_encoder.encode_source_batch(source_random_batch)

    # update state discriminator
    new_state_disc, state_disc_info, state_disc_stats_info = new_domain_encoder.state_discriminator.update(
        fake_batch=target_expert_batch["observations"],
        real_batch=source_expert_batch["observations"],
    )

    # update policy discriminator
    ## construct pairs
    target_random_pairs = get_state_pairs(target_random_batch)
    source_random_pairs = get_state_pairs(source_random_batch)
    source_expert_pairs = get_state_pairs(source_expert_batch)

    ## update
    new_policy_disc, policy_disc_info, policy_disc_stats_info = new_domain_encoder.policy_discriminator.update(
        fake_batch=jnp.concatenate([target_random_pairs, source_random_pairs]),
        real_batch=jnp.concatenate([source_expert_pairs, source_expert_pairs]), # TODO: jax-based crutch for Gradient Penalty usage
    )

    # update domain encoder
    new_domain_encoder = new_domain_encoder.replace(
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
