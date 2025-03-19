from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import wandb
from agents.imitation_learning.in_domain.gail.gail_discriminator.reward_transforms import \
    BaseRewardTransform
from agents.imitation_learning.utils import prepare_buffer
from misc.enot import ENOT
from misc.enot.utils import mapping_scatter
from utils import (SaveLoadFrozenDataclassMixin, convert_figure_to_array,
                   sample_batch_jit)
from utils.custom_types import Buffer, BufferState, DataType, PRNGKey

from .ot_buffer_factory import OTBufferFactory


class GWILENOT(PyTreeNode, SaveLoadFrozenDataclassMixin):
    rng: PRNGKey
    enot: ENOT
    ot_buffer: Buffer = struct.field(pytree_node=False)
    ot_target_buffer_state: BufferState = struct.field(pytree_node=False)
    source_expert_buffer: Buffer = struct.field(pytree_node=False)
    source_expert_buffer_state: BufferState = struct.field(pytree_node=False)
    reward_transform: BaseRewardTransform
    process_dict_batch_fn: Callable = struct.field(pytree_node=False)
    get_state_mapping: Callable = struct.field(pytree_node=False)
    _save_attrs: tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        enot_config: DictConfig,
        source_dim: int,
        #
        source_expert_buffer_state_path: str,
        batch_size: Optional[int],
        sourse_buffer_processor_config: Optional[DictConfig] = None,
        #
        ot_buffer_factory_config: Optional[DictConfig] = None,
        #
        reward_transform_config: DictConfig = None,
        use_pairs: bool = False,
        **kwargs,
    ):
        # expert buffer init
        source_expert_buffer, source_expert_buffer_state = prepare_buffer(
            buffer_state_path=source_expert_buffer_state_path,
            batch_size=batch_size,
            sourse_buffer_processor_config=sourse_buffer_processor_config,
        )
        target_dim = source_expert_buffer_state.experience["observations"].shape[-1]

        if ot_buffer_factory_config is not None:
            ot_buffer_factory = instantiate(ot_buffer_factory_config)
        else:
            ot_buffer_factory = OTBufferFactory()
        ot_buffer, ot_target_buffer_state = ot_buffer_factory(
            expert_state=source_expert_buffer_state,
            batch_size=batch_size,
        )

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
            rng=jax.random.key(seed),
            enot=enot,
            ot_buffer=ot_buffer,
            ot_target_buffer_state=ot_target_buffer_state,
            source_expert_buffer=source_expert_buffer,
            source_expert_buffer_state=source_expert_buffer_state,
            reward_transform=reward_transform,
            _save_attrs=("enot", "reward_transform"),
            process_dict_batch_fn=process_dict_batch_fn,
            get_state_mapping=get_state_mapping,
            **kwargs,
        )

    def encode_state(self, batch: DataType) -> DataType:
        batch = self.process_dict_batch_fn(batch)
        batch_mapped = self.enot(batch)
        return self.get_state_mapping(batch_mapped)


    @jax.jit
    def update(self, *, target_expert_batch: DataType): 
        # sample expert batch
        new_rng, source_expert_batch = sample_batch_jit(
            self.rng, self.ot_buffer, self.ot_target_buffer_state
        )

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

        self = self.replace(rng=new_rng, enot=new_enot, reward_transform=new_reward_transform)
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
        return -self.enot.cost(source, target_hat) + self.enot.g_potential_val(target_hat)

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
