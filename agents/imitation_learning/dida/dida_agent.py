import jax
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.imitation_learning.gail import GAILAgent
from utils.types import DataType

from .domain_encoder.base_domain_encoder import BaseDomainEncoder


class DIDAAgent(GAILAgent):
    domain_encoder: BaseDomainEncoder
    das: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        #
        encoding_dim: int,
        #
        expert_batch_size: int,
        expert_buffer_state_path: str,
        #
        domain_encoder_config: DictConfig,
        #
        das_config: DictConfig = None,
        #
        expert_buffer_state_processor_config: DictConfig = None,
        **kwargs,
    ):
        # domain encoder init
        domain_encoder = instantiate(
            domain_encoder_config,
            seed=seed,
            encoding_dim=encoding_dim,
            source_buffer_state_processor_config=expert_buffer_state_processor_config,
            _recursive_=False,
        )

        # DAS init
        das = None
        if das_config is not None:
            das = instantiate(das_config)

        # set attrs to save
        _save_attrs = kwargs.pop(
            "_save_attrs",
            ("agent", "policy_discriminator", "domain_encoder")
        )

        kwargs["observation_dim"] = encoding_dim
        return super().create(
            seed=seed,
            expert_batch_size=expert_batch_size,
            expert_buffer_state_path=expert_buffer_state_path,
            expert_buffer_state_processor_config=expert_buffer_state_processor_config,
            #
            domain_encoder=domain_encoder,
            das=das,
            _save_attrs=_save_attrs,
            **kwargs,
        )

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_target_state(observations)

    def _preprocess_expert_observations(self, observations: np.ndarray) -> np.ndarray:
        return self.domain_encoder.encode_source_state(observations)

    def update(self, batch: DataType):
        new_dida_agent, info, stats_info = _update_jit(
            dida_agent=self,
            learner_batch=batch,
        )
        return new_dida_agent, info, stats_info

def _update_jit(dida_agent: DIDAAgent, learner_batch: DataType):
    return _update_no_das_jit(
        dida_agent=dida_agent,
        learner_batch=learner_batch,
    )

@jax.jit
def _update_no_das_jit(dida_agent: DIDAAgent, learner_batch: DataType):
    # update domain encoder
    (
        new_domain_encoder,
        target_random_batch,
        target_expert_batch,
        source_random_batch,
        source_expert_batch,
        info,
        stats_info,
    ) = dida_agent.domain_encoder.update(
        target_expert_batch=learner_batch,
    )
    new_dida_agent = dida_agent.replace(domain_encoder=new_domain_encoder)

    # update agent and policy discriminator
    new_dida_agent, gail_info, gail_stats_info = new_dida_agent.update_gail(
        learner_batch=target_expert_batch,
        expert_batch=source_expert_batch,
        policy_discriminator_learner_batch=target_expert_batch,
    )

    info.update(gail_info)
    stats_info.update(gail_stats_info)
    return new_dida_agent, info, stats_info
