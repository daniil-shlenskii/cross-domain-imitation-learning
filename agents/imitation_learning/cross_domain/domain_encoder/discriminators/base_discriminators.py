from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing_extensions import override

from agents.imitation_learning.utils import get_state_pairs
from misc.gan.discriminator import LoosyDiscriminator
from utils import SaveLoadFrozenDataclassMixin
from utils.custom_types import DataType


class BaseDomainEncoderDiscriminators(PyTreeNode, SaveLoadFrozenDataclassMixin):
    state_discriminator: LoosyDiscriminator
    policy_discriminator: LoosyDiscriminator
    update_policy_discriminator_every: str = struct.field(pytree_node=False)
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        encoding_dim: int,
        state_discriminator_config: DictConfig,
        policy_discriminator_config: DictConfig,
        update_policy_discriminator_every: int = 1,
        **kwargs,
    ):
        state_discriminator = instantiate(
            state_discriminator_config,
            seed=seed,
            input_dim=encoding_dim,
            info_key="domain_encoder/state_discriminator",
            _recursive_=False,
        )
        policy_discriminator = instantiate(
            policy_discriminator_config,
            seed=seed,
            input_dim=encoding_dim * 2,
            info_key="domain_encoder/policy_discriminator",
            _recursive_=False,
        )
        return cls(
            state_discriminator=state_discriminator,
            policy_discriminator=policy_discriminator,
            update_policy_discriminator_every=update_policy_discriminator_every,
            _save_attrs=("state_discriminator", "policy_discriminator"),
            **kwargs,
        )

    @override
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
    discriminators: BaseDomainEncoderDiscriminators,
    target_random_batch: DataType,
    source_random_batch: DataType,
    source_expert_batch: DataType,
):
    # update state discriminator
    new_state_disc, state_disc_info, state_disc_stats_info = discriminators.state_discriminator.update(
        fake_batch=target_random_batch["observations"],
        real_batch=source_expert_batch["observations"],
    )

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

    # final update
    new_discriminators = discriminators.replace(
        state_discriminator=new_state_disc,
        policy_discriminator=new_policy_disc,
    )
    info = {**state_disc_info, **policy_disc_info}
    stats_info = {**state_disc_stats_info, **policy_disc_stats_info}

    return new_discriminators, info, stats_info
