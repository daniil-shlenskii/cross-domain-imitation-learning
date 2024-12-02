from typing import Tuple

import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.gail.reward_transforms import (IdentityRewardTransform,
                                           RewardTransform)
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils.types import DataType


class GAILDiscriminator(Discriminator):
    reward_transform: RewardTransform

    @classmethod
    def create(
        cls,
        *,
        reward_transform_config: DictConfig = None,
        **discriminator_kwargs,
    ):
        if reward_transform_config is not None:
            reward_transform = instantiate(reward_transform_config)
        else:
            reward_transform = IdentityRewardTransform.create()
        return super().create(
            reward_transform=reward_transform,
            _save_attrs=("state", "reward_transform"),
            **discriminator_kwargs
        )

    def update(self, *, expert_batch: DataType, learner_batch: DataType): 
        new_gail_discriminator, info, stats_info = _update_jit(
            expert_batch=expert_batch,
            learner_batch=learner_batch,
            gail_discriminator=self,
        )
        return new_gail_discriminator, info, stats_info
    
    def get_rewards(self, learner_batch: jnp.ndarray) -> jnp.ndarray:
        learner_state_pairs = jnp.concatenate([
            learner_batch["observations"],
            learner_batch["observations_next"]
        ], axis=1)
        return _get_rewards_jit(
            discriminator=self,
            learner_state_pairs=learner_state_pairs,
            reward_transform=self.reward_transform,
        )

@jax.jit
def _update_jit(
    expert_batch: DataType,
    learner_batch: DataType,
    gail_discriminator: GAILDiscriminator,
):
    # prepare batches
    expert_state_pairs = jnp.concatenate([
        expert_batch["observations"],
        expert_batch["observations_next"]
    ], axis=1)
    learner_state_pairs = jnp.concatenate([
        learner_batch["observations"],
        learner_batch["observations_next"]
    ], axis=1)
    
    # update discriminator
    new_gail_discrimiantor, info, stats_info = Discriminator.update(
        self=gail_discriminator,
        real_batch=expert_state_pairs,
        fake_batch=learner_state_pairs
    )

    # update reward transform
    base_rewards = _get_base_rewards(discriminator, learner_state_pairs)
    new_reward_transform, info = reward_transform.update(base_rewards)

    new_gail_discrimiantor = new_gail_discrimiantor.replace(reward_transform=new_reward_transform)
    info.update(reward_transform_info)
    return new_gail_discrimiantor, info, stats_info

@jax.jit
def _get_rewards_jit(
    discriminator: Discriminator,
    learner_state_pairs: jnp.ndarray,
    reward_transform: RewardTransform,
):
    base_rewards = _get_base_rewards(discriminator, learner_state_pairs)
    return reward_transform.transform(base_rewards)

def _get_base_rewards(
    discriminator: Discriminator,
    learner_state_pairs: jnp.ndarray,
):
    learner_logits = discriminator(learner_state_pairs)
    base_rewards = learner_logits
    return base_rewards
