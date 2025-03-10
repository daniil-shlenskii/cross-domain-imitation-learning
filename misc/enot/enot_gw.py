import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from utils import convert_figure_to_array

from .batch_preprocessors import BatchPreprocessor, IdentityPreprocessor
from .enot import ENOT
from .utils import mapping_scatter


class ENOTGW(ENOT):
    source_batch_preprocessor: BatchPreprocessor 
    target_batch_preprocessor: BatchPreprocessor 

    @classmethod
    def create(
        cls,
        source_dim: int,
        target_dim: int,
        source_batch_preprocessor_config: DictConfig=None,
        target_batch_preprocessor_config: DictConfig=None,
        **kwargs,
    ):
        kwargs["cost_fn_config"]["source_dim"] = source_dim
        kwargs["cost_fn_config"]["target_dim"] = target_dim
        if kwargs.get("train_cost_fn_config"):
            kwargs["train_cost_fn_config"]["source_dim"] = source_dim
            kwargs["train_cost_fn_config"]["target_dim"] = target_dim

        if source_batch_preprocessor_config is not None:
            source_batch_preprocessor = instantiate(source_batch_preprocessor_config, dim=source_dim)
        else:
            source_batch_preprocessor = IdentityPreprocessor.create()

        if target_batch_preprocessor_config is not None:
            target_batch_preprocessor = instantiate(target_batch_preprocessor_config, dim=target_dim)
        else:
            target_batch_preprocessor = IdentityPreprocessor.create()

        return super().create(
            source_dim=source_dim,
            target_dim=target_dim,
            source_batch_preprocessor=source_batch_preprocessor,
            target_batch_preprocessor=target_batch_preprocessor,
            **kwargs
        )

    @jax.jit
    def __call__(self, source: jnp.ndarray):
        source_encoded = self.source_batch_preprocessor.encode(source)
        target_hat_encoded = self.transport(source_encoded)
        target_hat = self.target_batch_preprocessor.decode(target_hat_encoded)
        return target_hat

    def cost(self, source: jnp.ndarray, target: jnp.ndarray):
        return jax.vmap(self.cost_fn)(
           self.source_batch_preprocessor.encode(source),
           self.target_batch_preprocessor.encode(target)
        )

    def g_potential_val(self, target: jnp.ndarray):
        return self.g_potential(self.target_batch_preprocessor.encode(target))

    @jax.jit
    def update(self, target: jnp.ndarray, source: jnp.ndarray):
        # preprocess batches
        new_source_batch_preprocessor, source, _ = self.source_batch_preprocessor.update(source)
        new_target_batch_preprocessor, target, _ = self.target_batch_preprocessor.update(target)

        # update cost fn
        target_hat = self.transport(source)
        new_cost_fn = self.cost_fn.update(source=source, target=target_hat)
        new_train_cost_fn = self.train_cost_fn.update(source=source, target=target_hat)
        self = self.replace(
            target_batch_preprocessor=new_target_batch_preprocessor,
            source_batch_preprocessor=new_source_batch_preprocessor,
            cost_fn=new_cost_fn,
            train_cost_fn=new_train_cost_fn
        )

        enot, info, stats_info = super().update(target, source)

        return enot, info, stats_info

    def evaluate(self, source: jnp.ndarray, target: jnp.ndarray, convert_to_wandb_type: bool=True):
        eval_info = super().evaluate(source, target, convert_to_wandb_type=convert_to_wandb_type)

        P = self.cost_fn.proj_matrix
        print(f"{P = }")
        source_enocoded = self.source_batch_preprocessor.encode(source)
        P_source_encoded = jax.vmap(lambda x: P @ x)(source_enocoded)
        P_source = self.target_batch_preprocessor.decode(P_source_encoded)
        figP = mapping_scatter(source, P_source, target)
        if convert_to_wandb_type:
            figP = wandb.Image(convert_figure_to_array(figP))

        eval_info["Pmapping"] = figP

        return eval_info 
