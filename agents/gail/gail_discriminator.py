from typing import Tuple

import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from gan.discriminator import Discriminator


class GAILDiscriminator(Discriminator):
    reward_transform: ...

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
        
        base_rewards = -jnp.log(_infer_discriminator(self, learner_batch))
        new_reward_transform = self.reward_transform.update(base_rewards)

        return new_state.replace(reward_transform=new_reward_transform), info, stats_info
    
    def get_rewards(self, x: jnp.ndarray) -> jnp.ndarray:
        rewards = -jnp.log(_infer_discriminator(self, x))
        return self.reward_transform.transform(rewards)

@jax.jit
def _infer_discriminator(discriminator: Discriminator, x: jnp.ndarray) -> jnp.ndarray:
    return discriminator(x)