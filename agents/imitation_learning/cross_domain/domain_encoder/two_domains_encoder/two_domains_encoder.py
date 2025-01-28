import jax
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf

from agents.imitation_learning.cross_domain.domain_encoder.base_domain_encoder import \
    BaseDomainEncoder
from misc.gan.generator import Generator
from utils.custom_types import DataType


class TwoDomainsEncoder(BaseDomainEncoder):
    source_encoder: Generator
    freeze_target_encoder: bool = struct.field(pytree_node=False)
    freeze_source_encoder: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        encoding_dim: int,
        source_encoder_config: DictConfig,
        freeze_target_encoder: bool = False,
        freeze_source_encoder: bool = False,
        **kwargs,
    ):
        # base domain encoder init
        base_domain_encoder = super().create(
            seed=seed,
            encoding_dim=encoding_dim,
            source_encoder=None,
            freeze_target_encoder=freeze_target_encoder,
            freeze_source_encoder=freeze_source_encoder,
            **kwargs,
        )

        # source encoder init
        source_encoder_config = OmegaConf.to_container(source_encoder_config)
        source_encoder_config["loss_config"]["state_loss"] = base_domain_encoder.state_discriminator.state.loss_fn
        source_encoder_config["loss_config"]["policy_loss"] = base_domain_encoder.policy_discriminator.state.loss_fn

        source_dim = base_domain_encoder.source_random_buffer_state.experience["observations"].shape[-1]

        source_encoder = instantiate(
            source_encoder_config,
            seed=seed,
            input_dim=source_dim,
            output_dim=encoding_dim,
            info_key="source_encoder",
            _recursive_=False,
        )

        # update _save_attr
        _save_attrs = base_domain_encoder._save_attrs + ("source_encoder",)

        return base_domain_encoder.replace(
            source_encoder=source_encoder,
            _save_attrs=_save_attrs,
        )

    def _update_encoder(
        self,
        *,
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
   ):
        new_target_encoder, target_info, target_stats_info = self.target_encoder.update(
            target_random_batch=target_random_batch,
            policy_discriminator=self.policy_discriminator,
            state_discriminator=self.state_discriminator,
        )
        new_target_encoder = jax.lax.cond(
            self.freeze_target_encoder,
            lambda: self.target_encoder,
            lambda: new_target_encoder,
        )

        new_source_encoder, source_info, source_stats_info = self.source_encoder.update(
            source_random_batch=source_random_batch,
            source_expert_batch=source_expert_batch,
            policy_discriminator=self.policy_discriminator,
            state_discriminator=self.state_discriminator,
        )
        new_source_encoder = jax.lax.cond(
            self.freeze_source_encoder,
            lambda: self.source_encoder,
            lambda: new_source_encoder,
        )

        new_encoder = self.replace(
            target_encoder=new_target_encoder,
            source_encoder=new_source_encoder
        )
        info = {**target_info, **source_info}
        stats_info = {**target_stats_info, **source_stats_info}
        return new_encoder, info, stats_info
