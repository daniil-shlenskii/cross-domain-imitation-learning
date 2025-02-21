import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from misc.enot import ENOT
from misc.enot.utils import mapping_scatter
from utils.custom_types import DataType
from utils.utils import convert_figure_to_array

from .gail_discriminator import GAILDiscriminator


class GAILDiscriminatorWithENOT(GAILDiscriminator):
    enot: ENOT

    @classmethod
    def create(
        cls,
        *,
        input_dim: int,
        enot_config: DictConfig,
        **gail_discriminator_kwargs,
    ):
        enot = instantiate(enot_config, data_dim=input_dim//2, _recursive_=False)
        return super().create(
            input_dim=input_dim,
            enot=enot,
            _save_attrs=("state", "reward_transform", "enot"),
            **gail_discriminator_kwargs,
        )

    @jax.jit
    def update(self, *, target_expert_batch: DataType, source_expert_batch: DataType):
        new_enot, enot_info, enot_stats_info = self.enot.update(
            source=target_expert_batch,
            target=source_expert_batch,
        )

        target_expert_batch["observations"] = self.enot(target_expert_batch["observations"])
        target_expert_batch["observations_next"] = self.enot(target_expert_batch["observations_next"])

        new_gail_discr, discr_info, discrs_stats_info = super().update(
            target_expert_batch=target_expert_batch,
            source_expert_batch=source_expert_batch,
        )

        new_gail_discr = new_gail_discr.replace(enot=new_enot)
        info = {**discr_info, **enot_info}
        stats_info = {**discrs_stats_info, **enot_stats_info}
        return new_gail_discr, info, stats_info

    @jax.jit
    def get_rewards(self, target_expert_batch: DataType) -> jnp.ndarray:
        target_expert_batch["observations"] = self.enot(target_expert_batch["observations"])
        target_expert_batch["observations_next"] = self.enot(target_expert_batch["observations_next"])
        return super().get_rewards(target_expert_batch=target_expert_batch)

    def evaluate(self, traj_dict: dict, convert_to_wandb_type: bool=True):
        eval_info = super().evaluate(traj_dict=traj_dict, convert_to_wandb_type=convert_to_wandb_type)

        # enot visualization
        source = traj_dict["states"]["TE"]
        target_hat = self.enot(source)
        target = traj_dict["states"]["SE"]
        fig = mapping_scatter(source, target_hat, target)
        if convert_to_wandb_type:
            fig = wandb.Image(convert_figure_to_array(fig))
        eval_info[f"{self.state.info_key}/enot_mapping_scatter"] = fig

        return eval_info
