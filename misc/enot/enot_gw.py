import jax
import jax.numpy as jnp

import wandb
from utils import convert_figure_to_array

from .enot import ENOT
from .utils import mapping_scatter


class ENOTGW(ENOT):
    use_projection: bool

    @classmethod
    def create(
        cls,
        source_dim: int,
        target_dim: int,
        use_projection: bool=False,
        **kwargs,
    ):
        kwargs["cost_fn_config"]["source_dim"] = source_dim
        kwargs["cost_fn_config"]["target_dim"] = target_dim
        if kwargs.get("train_cost_fn_config"):
            kwargs["train_cost_fn_config"]["source_dim"] = source_dim
            kwargs["train_cost_fn_config"]["target_dim"] = target_dim

        enot = super().create(
            source_dim=source_dim,
            target_dim=target_dim,
            use_projection=use_projection,
            **kwargs
        )

        if use_projection:
            transport_loss = enot.transport.loss_fn
            def new_transport_loss(params, state, source, enot):
                proj_matrix = enot.train_cost_fn.proj_matrix
                source = jax.vmap(lambda x: proj_matrix @ x)(source)
                return transport_loss(params, state, source, enot)
            new_transport = enot.transport.replace(loss_fn=new_transport_loss)
            enot = enot.replace(transport=new_transport)
        return enot

    @jax.jit
    def __call__(self, source: jnp.ndarray):
        source_encoded = self.source_batch_preprocessor.encode(source)

        source_encoded = jax.lax.cond(
            self.use_projection,
            lambda x: jax.vmap(lambda x: self.train_cost_fn.proj_matrix @ x)(x),
            lambda x: x,
            source_encoded
        )

        target_hat_encoded = self.transport(source_encoded)
        target_hat = self.target_batch_preprocessor.decode(target_hat_encoded)
        return target_hat

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
