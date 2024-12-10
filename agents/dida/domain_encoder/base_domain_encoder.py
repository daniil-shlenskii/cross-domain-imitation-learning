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


class BaseDomainEncoder(PyTreeNode, SaveLoadFrozenDataclassMixin, ABC):
    learner_encoder: Generator
    state_discriminator: Discriminator
    policy_discriminator: Discriminator
    state_loss_scale: float
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
        state_discriminator_config: DictConfig,
        policy_discriminator_config: DictConfig,
        #
        state_loss_scale: float,
        #
        **kwargs,
    ):
        # state discriminator init
        state_discriminator = instantiate(
            state_discriminator_config,
            seed=seed,
            input_dim=encoding_dim * 2,
            info_key="state_discriminator",
            _recursive_=False,
        )

        # policy discriminator init
        policy_discriminator = instantiate(
            policy_discriminator_config,
            seed=seed,
            input_dim=encoding_dim * 2,
            info_key="policy_discriminator",
            _recursive_=False,
        )

        # learner encoder init
        learner_encoder_config = OmegaConf.to_container(learner_encoder_config)
        learner_encoder_config["loss_config"]["state_loss"] = state_discriminator.state.loss_fn
        learner_encoder_config["loss_config"]["policy_loss"] = policy_discriminator.state.loss_fn

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
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            state_loss_scale=state_loss_scale,
            _save_attrs=(
                "learner_encoder",
                "state_discriminator",
                "policy_discriminator"
            ),
            **kwargs,
        )

    @jax.jit 
    def encode_learner_state(self, state: jnp.ndarray):
        return self.learner_encoder(state)

    @jax.jit
    def encode_expert_state(self, state: jnp.ndarray):
        return self.learner_encoder(state)

    def update(
        self,
        learner_batch: DataType,
        expert_batch: DataType,
        anchor_batch: DataType,
    ):
        (
            new_domain_encoder,
            learner_batch,
            expert_batch,
            anchor_batch,
            learner_domain_logits,
            expert_domain_logits,
            info,
            stats_info,
        ) = _update_jit(
            domain_encoder=self,
            learner_batch=learner_batch,
            expert_batch=expert_batch,
            anchor_batch=anchor_batch,
        )
        return (
            new_domain_encoder,
            learner_batch,
            expert_batch,
            anchor_batch,
            learner_domain_logits,
            expert_domain_logits,
            info,
            stats_info,
        )

    @abstractmethod
    def _update_encoder(
        self,
        learner_batch: DataType,
        expert_batch: DataType,
        anchor_batch: DataType,
   ):
        pass

@jax.jit
def _update_jit(
    domain_encoder: BaseDomainEncoder,
    learner_batch: DataType,
    expert_batch: DataType,
    anchor_batch: DataType,
):
    # update encoder 
    new_domain_encoder, info, stats_info = domain_encoder._update_encoder(
        learner_batch=learner_batch,
        expert_batch=expert_batch,
    )
    learner_batch = info.pop("learner_encoder_encoded_batch")
    expert_batch = info.pop("expert_encoder_encoded_batch")

    # encode anchor batch
    anchor_batch["observations"] = new_domain_encoder.encode_expert_state(anchor_batch["observations"])
    anchor_batch["observations_next"] = new_domain_encoder.encode_expert_state(anchor_batch["observations_next"])

    # construct pairs
    learner_pairs = get_state_pairs(learner_batch)
    expert_pairs = get_state_pairs(expert_batch)
    anchor_pairs = get_state_pairs(anchor_batch)

    # update state discriminator
    new_state_disc, state_disc_info, state_disc_stats_info = new_domain_encoder.state_discriminator.update(
        fake_batch=learner_pairs,
        real_batch=jnp.concatenate([expert_pairs, anchor_pairs]),
        return_logits=True,
    )
    learner_domain_logits = state_disc_info.pop("fake_logits")
    expert_domain_logits = state_disc_info.pop("real_logits")

    # update policy discriminator
    new_policy_disc, policy_disc_info, policy_disc_stats_info = new_domain_encoder.policy_discriminator.update(
        fake_batch=jnp.concatenate([learner_pairs, anchor_pairs]),
        real_batch=expert_pairs,
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
        learner_batch,
        expert_batch,
        anchor_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info,
    )
