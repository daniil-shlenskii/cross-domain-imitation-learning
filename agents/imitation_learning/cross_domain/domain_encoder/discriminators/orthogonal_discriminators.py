from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing_extensions import override

from agents.imitation_learning.cross_domain.domain_encoder.discriminators.base_discriminators import \
    BaseDomainEncoderDiscriminators
from agents.imitation_learning.utils import get_state_pairs
from misc.gan.discriminator import LoosyDiscriminator
from nn.train_state import _compute_norms
from utils import SaveLoadFrozenDataclassMixin
from utils.custom_types import DataType


class OrthogonalDomainEncoderDiscriminators(BaseDomainEncoderDiscriminators):
    loss_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        loss_fn_config: DictConfig,
        **kwargs,
    ):
        return super().create(
            loss_fn=instantiate(loss_fn_config),
            **kwargs,
        )

    def update(
        self,
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        return _update_jit(
            discriminators=self,
            target_random_batch=target_random_batch,
            source_random_batch=source_random_batch,
            source_expert_batch=source_expert_batch,
        )

@jax.jit
def _update_jit(
    discriminators: OrthogonalDomainEncoderDiscriminators,
    target_random_batch: DataType,
    source_random_batch: DataType,
    source_expert_batch: DataType,
):
    #
    state_discr_state = discriminators.state_discriminator.state
    policy_discr_state = discriminators.policy_discriminator.state

    # get grads
    (_, info), (state_discr_grad, policy_discr_grad) = jax.value_and_grad(
        discriminators.loss_fn, argnums=(0, 1), has_aux=True
    )(
        state_discr_state.params,
        policy_discr_state.params,
        state_discriminator_state=state_discr_state,
        policy_discriminator_state=policy_discr_state,
        target_random_batch=target_random_batch,
        source_random_batch=source_random_batch,
        source_expert_batch=source_expert_batch,
    )

    # get stats info
    stats_info = {}
    stats_info[f"{state_discr_state.info_key}/max_grad_norm"] = _compute_norms(state_discr_grad)
    stats_info[f"{state_discr_state.info_key}/max_weight_norm"] = _compute_norms(state_discr_state.params)

    stats_info[f"{policy_discr_state.info_key}/max_grad_norm"] = _compute_norms(policy_discr_grad)
    stats_info[f"{policy_discr_state.info_key}/max_weight_norm"] = _compute_norms(policy_discr_state.params)

    # update
    new_state_discr_state = state_discr_state.apply_gradients(grads=state_discr_grad)
    new_policy_discr_state = policy_discr_state.apply_gradients(grads=policy_discr_grad)
    # new_policy_discr_state = jax.lax.cond(
    #     (discriminators.state_discriminator.state.step + 1) % discriminators.update_policy_discriminator_every == 0,
    #     lambda: policy_discr_state.apply_gradients(grads=policy_discr_grad),
    #     lambda: policy_discr_state,
    # )

    new_state_discr = discriminators.state_discriminator.replace(state=new_state_discr_state)
    new_policy_discr = discriminators.policy_discriminator.replace(state=new_policy_discr_state)

    new_discriminators = discriminators.replace(
        state_discriminator=new_state_discr,
        policy_discriminator=new_policy_discr,
    )

    return new_discriminators, info, stats_info
