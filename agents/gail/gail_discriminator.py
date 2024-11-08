from typing import Tuple

import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.gail.reward_transforms import RewardTransform
from gan.discriminator import Discriminator


class GAILDiscriminator(Discriminator):
    reward_transform: RewardTransform

    @classmethod
    def create(
        cls,
        *,
        reward_transform_config: DictConfig,
        **discriminator_kwargs,
    ):
        reward_transform = instantiate(reward_transform_config)
        return super().create(
            reward_transform=reward_transform,
            _save_attrs=("state", "reward_transform"),
            **discriminator_kwargs
        )

    def update(self, *, expert_batch: jnp.ndarray, learner_batch: jnp.ndarray):
        new_state, info, stats_info = super().update(real_batch=expert_batch, fake_batch=learner_batch)
        new_reward_transform = _update_reward_transform(
            learner_batch=learner_batch,
            discriminator=new_state,
            reward_transform=self.reward_transform
        )
        return new_state.replace(reward_transform=new_reward_transform), info, stats_info
    
    def get_rewards(self, x: jnp.ndarray) -> jnp.ndarray:
        return _get_rewards_jit(
            x,
            discriminator=self,
            reward_transform=self.reward_transform
        )

@jax.jit
def _update_reward_transform(
    learner_batch: jnp.ndarray,
    discriminator: Discriminator,
    reward_transform: RewardTransform,
):  
    base_rewards = -jnp.log(discriminator(learner_batch))
    new_reward_transform = reward_transform.update(base_rewards)
    return new_reward_transform

@jax.jit
def _get_rewards_jit(x: jnp.ndarray, discriminator: Discriminator, reward_transform: RewardTransform):
    rewards = -jnp.log(discriminator(x))
    return reward_transform.transform(rewards)