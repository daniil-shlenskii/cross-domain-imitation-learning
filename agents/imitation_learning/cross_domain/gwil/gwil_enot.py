from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import wandb
from agents.imitation_learning.in_domain.gail.gail_discriminator.reward_transforms import \
    BaseRewardTransform
from misc.enot import ENOT
from misc.enot.utils import mapping_scatter
from utils import SaveLoadFrozenDataclassMixin, convert_figure_to_array
from utils.custom_types import DataType


class GWILENOT(PyTreeNode, SaveLoadFrozenDataclassMixin):
    enot: ENOT
    reward_transform: BaseRewardTransform
    process_dict_batch_fn: Callable = struct.field(pytree_node=False)
    get_state_mapping: Callable = struct.field(pytree_node=False)
    _save_attrs: tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        enot_config: DictConfig,
        source_dim: int,
        target_dim: int,
        reward_transform_config: DictConfig = None,
        use_pairs: bool = False,
    ):
        if reward_transform_config is not None:
            reward_transform = instantiate(reward_transform_config)
        else:
            reward_transform = BaseRewardTransform.create()

        if use_pairs:
            def get_state_mapping(batch_mapped):
                return jnp.split(batch_mapped, 2, axis=-1)[0]
            def process_dict_batch_fn(dict_batch):
                return jnp.concatenate([
                    dict_batch["observations"], dict_batch["observations_next"]
                ], axis=1)
            source_dim *= 2
            target_dim *= 2
        else:
            def get_state_mapping(batch_mapped):
                return batch_mapped
            def process_dict_batch_fn(dict_batch):
                return dict_batch["observations"]

        enot = instantiate(
            enot_config,
            source_dim=source_dim,
            target_dim=target_dim,
            _recursive_=False,
        )

        return cls(
            enot=enot,
            reward_transform=reward_transform,
            _save_attrs=("enot", "reward_transform"),
            process_dict_batch_fn=process_dict_batch_fn,
            get_state_mapping=get_state_mapping,
        )

    def encode_state(self, batch: DataType) -> DataType:
        batch = self.process_dict_batch_fn(batch)
        batch_mapped = self.enot(batch)
        return self.get_state_mapping(batch_mapped)


    @jax.jit
    def update(self, *, target_expert_batch: DataType, source_expert_batch: DataType): 
        # process dict batch
        target_expert_batch = self.process_dict_batch_fn(target_expert_batch)
        source_expert_batch = self.process_dict_batch_fn(source_expert_batch)

        # update enot
        new_enot, enot_info, enot_stats_info = self.enot.update(
            source=target_expert_batch, target=source_expert_batch,
        )

        # update reward transform
        base_rewards = self.get_base_rewards(target_expert_batch)
        new_reward_transform, reward_transform_info = self.reward_transform.update(base_rewards)

        self = self.replace(enot=new_enot, reward_transform=new_reward_transform)
        info = {**enot_info, **reward_transform_info}
        stats_info = {**enot_stats_info}
        return self, info, stats_info

    @jax.jit
    def get_rewards(self, target_expert_batch: DataType) -> jnp.ndarray:
        target_expert_batch = self.process_dict_batch_fn(target_expert_batch)
        base_rewards = self.get_base_rewards(target_expert_batch)
        return self.reward_transform.transform(base_rewards)

    @jax.jit
    def get_base_rewards(self, target_expert_batch: jnp.ndarray) -> jnp.ndarray:
        source = target_expert_batch
        target_hat = self.enot(source)
        return self.enot.cost(source, target_hat)

    def evaluate(self, source_pairs: jnp.ndarray, target_pairs: jnp.ndarray, convert_to_wandb_type: bool=True):
        keys = ["observations", "observations_next"]
        source_batch  = dict(zip(keys, jnp.split(source_pairs, 2, axis=-1)))
        target_batch  = dict(zip(keys, jnp.split(target_pairs, 2, axis=-1)))

        source = self.process_dict_batch_fn(source_batch)
        target = self.process_dict_batch_fn(target_batch)
        target_hat = self.enot(source)

        source_state = self.get_state_mapping(source)
        target_state = self.get_state_mapping(target)
        target_hat_state = self.get_state_mapping(target_hat)

        fig = mapping_scatter(source_state, target_hat_state, target_state)
        if convert_to_wandb_type:
            fig = wandb.Image(convert_figure_to_array(fig), caption="")

        return {"enot/mapping_scatter": fig}
