import jax
import jax.numpy as jnp

from misc.enot.enot_gw import ENOTGW
from nn.train_state import TrainState
from utils.custom_types import Params


class ContinuityRegularizedTransportLoss:
    def __init__(self, reg_scale: float = 1.):
        self.reg_scale = reg_scale

    def __call__(
        self,
        params: Params,
        state: TrainState,
        source: jnp.ndarray,
        source_next: jnp.ndarray,
        enot,
    ):
        cost_fn = jax.vmap(enot.train_cost_fn)

        target_hat = state.apply_fn({"params": params}, source)
        target_hat_next = state.apply_fn({"params": params}, source_next)

        downstream_loss = (cost_fn(source, target_hat) - enot.g_potential(target_hat)).mean()
        continuity_loss = ((target_hat_next - target_hat)**2).sum(axis=1).mean()
        loss = downstream_loss + self.reg_scale * continuity_loss

        return loss, {
            f"{state.info_key}/loss": loss,
            f"{state.info_key}/downstream_loss": downstream_loss,
            f"{state.info_key}/continuity_loss": continuity_loss,
            "target_hat": target_hat,
        }

class ENOTGWContinuous(ENOTGW):
    @jax.jit
    def update(self, target: jnp.ndarray, source: jnp.ndarray, source_next: jnp.ndarray):
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

        # update transport and g_potential
        new_transport, new_transport_info, new_transport_stats_info = self.transport.update(
            source=source,
            source_next=source_next,
            enot=self,
        )
        target_hat = new_transport_info.pop("target_hat")
        new_g_potential, new_g_potential_info, new_g_potential_stats_info = self.g_potential.update(
            source=source,
            target=target,
            target_hat=target_hat,
            enot=self,
        )
        new_self = self.replace(
            transport=new_transport,
            g_potential=new_g_potential,
        )
        info = {**new_transport_info, **new_g_potential_info}
        stats_info = {**new_transport_stats_info, **new_g_potential_stats_info}
        return new_self, info, stats_info
