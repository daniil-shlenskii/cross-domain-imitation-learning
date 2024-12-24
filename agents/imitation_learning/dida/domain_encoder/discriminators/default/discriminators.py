import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig

from agents.imitation_learning.dida.domain_encoder.discriminators.interface import \
    BaseDomainEncoderDiscriminators
from agents.imitation_learning.utils import get_state_pairs
from gan.discriminator import Discriminator
from utils.types import DataType


class DomainEncoderDiscriminators(BaseDomainEncoderDiscriminators):
    state_discriminator: Discriminator
    policy_discriminator: Discriminator

    @classmethod
    def create(
        cls,
        seed: int,
        encoding_dim: int,
        state_discriminator_config: DictConfig,
        policy_discriminator_config: DictConfig,
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
            _save_attrs=("state_discriminator", "policy_discriminator")
        )

    def get_state_discriminator(self):
        return self.state_discriminator

    def get_policy_discriminator(self):
        return self.policy_discriminator

    def get_state_loss(self):
        return self.state_discriminator.state.loss_fn

    def get_policy_loss(self):
        return self.policy_discriminator.state.loss_fn

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
    discriminators: DomainEncoderDiscriminators,
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

    # final update
    new_discriminators = discriminators.replace(
        state_discriminator=new_state_disc,
        policy_discriminator=new_policy_disc, 
    )
    info = {**state_disc_info, **policy_disc_info}
    stats_info = {**state_disc_stats_info, **policy_disc_stats_info}

    return new_discriminators, info, stats_info
