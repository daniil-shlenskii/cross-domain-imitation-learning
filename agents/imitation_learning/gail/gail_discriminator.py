import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from gan.discriminator import Discriminator
from utils.types import DataType

from .reward_transforms import BaseRewardTransform


class GAILDiscriminator(Discriminator):
    neg_reward_factor: float
    reward_transform: BaseRewardTransform

    @classmethod
    def create(
        cls,
        *,
        neg_reward_factor: float = 1.,
        reward_transform_config: DictConfig = None,
        **discriminator_kwargs,
    ):
        if reward_transform_config is not None:
            reward_transform = instantiate(reward_transform_config)
        else:
            reward_transform = BaseRewardTransform.create()
        return super().create(
            neg_reward_factor=neg_reward_factor,
            reward_transform=reward_transform,
            info_key="policy_discriminator",
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
            gail_discriminator=self,
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
    new_gail_discr, gail_discr_info, gail_discr_stats_info = Discriminator.update(
        self=gail_discriminator,
        real_batch=expert_state_pairs,
        fake_batch=learner_state_pairs
    )

    # update reward transform
    base_rewards = _get_base_rewards(new_gail_discr, learner_state_pairs)
    new_reward_transform, reward_transform_info = new_gail_discr.reward_transform.update(base_rewards)

    new_gail_discr = new_gail_discr.replace(reward_transform=new_reward_transform)
    info = {**gail_discr_info, **reward_transform_info}
    stats_info = {**gail_discr_stats_info}
    return new_gail_discr, info, stats_info

@jax.jit
def _get_rewards_jit(
    gail_discriminator: Discriminator,
    learner_state_pairs: jnp.ndarray,
    reward_transform: BaseRewardTransform,
):
    base_rewards = _get_base_rewards(gail_discriminator, learner_state_pairs)
    return reward_transform.transform(base_rewards)

def _get_base_rewards(
    gail_discriminator: Discriminator,
    learner_state_pairs: jnp.ndarray,
):
    learner_logits = gail_discriminator(learner_state_pairs)
    base_rewards = learner_logits + jnp.clip(learner_logits, max=0.) * gail_discriminator.neg_reward_factor
    return base_rewards
