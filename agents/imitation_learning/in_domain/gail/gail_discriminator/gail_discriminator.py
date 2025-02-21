import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from typing_extensions import override

import wandb
from agents.imitation_learning.in_domain.gail.utils import \
    get_trajs_discriminator_logits_and_accuracy
from agents.imitation_learning.utils import get_state_pairs, prepare_buffer
from misc.gan.discriminator import Discriminator
from utils import convert_figure_to_array
from utils.custom_types import DataType

from .reward_transforms import BaseRewardTransform


class GAILDiscriminator(Discriminator):
    reward_transform: BaseRewardTransform

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
            reward_transform = BaseRewardTransform.create()
            
        _save_attrs = discriminator_kwargs.pop("_save_attrs", ("state", "reward_transform"))

        return super().create(
            reward_transform=reward_transform,
            info_key="policy_discriminator",
            _save_attrs=_save_attrs,
            **discriminator_kwargs
        )

    def update(self, *, target_expert_batch: DataType, source_expert_batch: DataType, ): 
        new_gail_discriminator, info, stats_info = _update_jit(
            target_expert_batch=target_expert_batch,
            source_expert_batch=source_expert_batch,
            gail_discriminator=self,
        )
        return new_gail_discriminator, info, stats_info

    def get_rewards(self, target_expert_batch: DataType) -> jnp.ndarray:
        return _get_rewards_jit(
            gail_discriminator=self,
            target_expert_state_pairs=get_state_pairs(target_expert_batch),
            reward_transform=self.reward_transform,
        )

    @override
    def evaluate(self, traj_dict: dict, convert_to_wandb_type: bool=True):
        eval_info = {}

        # get logits plot and accuracy of policy_discriminator
        accuracy_dict, logits_figure_dict = get_trajs_discriminator_logits_and_accuracy(
            discriminator=self,
            traj_dict=traj_dict["state_pairs"],
            keys_to_use=["TE", "SE"],
            discriminator_key="policy",
        )
        if convert_to_wandb_type:
            for k in logits_figure_dict:
                logits_figure_dict[k] = wandb.Image(convert_figure_to_array(logits_figure_dict[k]), caption="")
        eval_info.update(**accuracy_dict, **logits_figure_dict)

        return eval_info

@jax.jit
def _update_jit(
    target_expert_batch: DataType,
    source_expert_batch: DataType,
    gail_discriminator: GAILDiscriminator,
):
    # prepare state pairs
    target_expert_state_pairs = get_state_pairs(target_expert_batch)
    source_expert_state_pairs = get_state_pairs(source_expert_batch)

    # update discriminator
    new_gail_discr, gail_discr_info, gail_discr_stats_info = Discriminator.update(
        self=gail_discriminator,
        fake_batch=target_expert_state_pairs,
        real_batch=source_expert_state_pairs,
    )

    # update reward transform
    base_rewards = _get_base_rewards(new_gail_discr, target_expert_state_pairs)
    new_reward_transform, reward_transform_info = new_gail_discr.reward_transform.update(base_rewards)

    new_gail_discr = new_gail_discr.replace(reward_transform=new_reward_transform)
    info = {**gail_discr_info, **reward_transform_info}
    stats_info = {**gail_discr_stats_info}
    return new_gail_discr, info, stats_info

@jax.jit
def _get_rewards_jit(
    gail_discriminator: Discriminator,
    target_expert_state_pairs: jnp.ndarray,
    reward_transform: BaseRewardTransform,
):
    base_rewards = _get_base_rewards(gail_discriminator, target_expert_state_pairs)
    return reward_transform.transform(base_rewards)

def _get_base_rewards(
    gail_discriminator: Discriminator,
    target_expert_state_pairs: jnp.ndarray,
):
    logits = gail_discriminator(target_expert_state_pairs)
    return logits
