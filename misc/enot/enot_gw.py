import jax
import jax.numpy as jnp

import wandb
from utils import convert_figure_to_array

from .enot import ENOT
from .utils import mapping_scatter


class ENOTGW(ENOT):

    @classmethod
    def create(
        cls,
        source_dim: int,
        target_dim: int,
        **kwargs,
    ):
        kwargs["cost_fn_config"]["source_dim"] = source_dim
        kwargs["cost_fn_config"]["target_dim"] = target_dim
        if kwargs.get("train_cost_fn_config"):
            kwargs["train_cost_fn_config"]["source_dim"] = source_dim
            kwargs["train_cost_fn_config"]["target_dim"] = target_dim

        return super().create(
            source_dim=source_dim,
            target_dim=target_dim,
            **kwargs
        )


    def evaluate(self, source: jnp.ndarray, target: jnp.ndarray, convert_to_wandb_type: bool=True):
        eval_info = super().evaluate(source, target, convert_to_wandb_type=convert_to_wandb_type)

        P = self.cost_fn.proj_matrix
        source_enocoded = self.source_batch_preprocessor.encode(source)
        P_source_encoded = jax.vmap(lambda x: P @ x)(source_enocoded)
        P_source = self.target_batch_preprocessor.decode(P_source_encoded)
        figP = mapping_scatter(source, P_source, target)
        if convert_to_wandb_type:
            figP = wandb.Image(convert_figure_to_array(figP))

        eval_info["Pmapping"] = figP

        return eval_info 
