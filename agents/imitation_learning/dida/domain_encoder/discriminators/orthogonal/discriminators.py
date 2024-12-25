from typing import Callable

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf import DictConfig

from agents.imitation_learning.dida.domain_encoder.discriminators.interface import \
    BaseDomainEncoderDiscriminators
from agents.imitation_learning.utils import get_state_pairs
from nn.train_state import TrainState
from utils.types import DataType
from utils.utils import instantiate_optimizer


class OrthogonalDomainEncoderDiscriminators(BaseDomainEncoderDiscriminators):
    state: TrainState

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        encoding_dim: int,
        module_config: DictConfig,
        loss_fn_config: DictConfig,
        optimizer_config: DictConfig,
    ):
        module = instantiate(module_config)
        params = module.init(jax.random.key(seed), jnp.ones(encoding_dim * 2))["params"]
        state = TrainState.create(
            loss_fn=instantiate(loss_fn_config),
            apply_fn=module.apply,
            params=params,
            tx=instantiate_optimizer(optimizer_config),
            info_key="discriminators",
        )
        return cls(
            state=state,
            has_state_discriminator_paired_input=True,
            _save_attrs=("state",)
        )

    def get_state_discriminator(self):
        class StateDiscriminator(PyTreeNode):
            discriminators_state: "OrthogonalDomainEncoderDiscriminators"
            def __call__(self, x: jnp.ndarray):
                return self.discriminators_state(x)[0]
        return StateDiscriminator(self.state)

    def get_policy_discriminator(self):
        class PolicyDiscriminator(PyTreeNode):
            discriminators_state: "OrthogonalDomainEncoderDiscriminators"
            def __call___(self, x: jnp.ndarray):
                return self.discriminators_state(x)[1]
        return PolicyDiscriminator(self.state)

    def get_state_loss(self):
        return self.state.loss_fn

    def get_policy_loss(self):
        return self.state.loss_fn

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
    target_random_pairs = get_state_pairs(target_random_batch)
    source_random_pairs = get_state_pairs(source_random_batch)
    source_expert_pairs = get_state_pairs(source_expert_batch)

    new_state, info, stats_info = discriminators.state.update(
        target_random_pairs=target_random_pairs,
        source_random_pairs=source_random_pairs,
        source_expert_pairs=source_expert_pairs,
    )

    new_discriminators = discriminators.replace(state=new_state)
    return new_discriminators, info, stats_info
