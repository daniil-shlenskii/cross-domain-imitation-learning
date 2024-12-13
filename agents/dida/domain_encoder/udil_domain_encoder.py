from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from agents.gail.utils import get_state_pairs
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils import SaveLoadFrozenDataclassMixin
from utils.types import DataType


class UDILDomainEncoder(PyTreeNode, SaveLoadFrozenDataclassMixin):
    learner_encoder: Generator
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        learner_dim: int,
        encoding_dim: int ,
        #
        learner_encoder_config: DictConfig,
        #
        **kwargs,
    ):
        learner_encoder = instantiate(
            learner_encoder_config,
            seed=seed,
            input_dim=learner_dim,
            output_dim=encoding_dim,
            info_key="learner_encoder",
            _recursive_=False,
        )
        return cls(
            learner_encoder=learner_encoder,
            _save_attrs=("learner_encoder",),
        )

    @jax.jit
    def encode_learner_state(self, state: jnp.ndarray):
        return self.learner_encoder(state)

    def encode_expert_state(self, state: jnp.ndarray):
        return state

    def update(
        self,
        learner_batch: DataType,
        expert_batch: DataType,
        anchor_batch: DataType,
        policy_discriminator: Discriminator,
    ):
        (
            new_domain_encoder,
            info,
            stats_info,
        ) = _update_jit(
            domain_encoder=self,
            learner_batch=learner_batch,
            policy_discriminator=policy_discriminator,
        )
        return (
            new_domain_encoder,
            learner_batch,
            expert_batch,
            anchor_batch,
            None,
            None,
            info,
            stats_info,
        )

@jax.jit
def _update_jit(
    domain_encoder: UDILDomainEncoder,
    learner_batch: DataType,
    policy_discriminator: Discriminator,
):
    new_learner_encoder, info, stats_info = domain_encoder.learner_encoder.update(
        batch=learner_batch,
        discriminator=policy_discriminator,
    )
    new_domain_encoder = domain_encoder.replace(
        learner_encoder=new_learner_encoder
    )
    return new_domain_encoder, info, stats_info

class LearnerEncoderLoss:
    def __call__(
        self,
        params,
        state,
        batch,
        discriminator,
    ):
        fake_state_pairs = jnp.concatenate([
            state.apply_fn({"params": params}, batch["observations"], train=True),
            state.apply_fn({"params": params}, batch["observations_next"], train=True),
        ], axis=1)
        fake_logits = discriminator(fake_state_pairs)
        loss = discriminator.state.loss_fn.generator_loss_fn(fake_logits)

        info = {f"{state.info_key}_loss": loss}
        return loss, info
