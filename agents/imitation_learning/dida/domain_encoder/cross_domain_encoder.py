from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf

from gan.generator import Generator
from utils.types import DataType

from .base_domain_encoder import BaseDomainEncoder


class CrossDomainEncoder(BaseDomainEncoder):
    source_encoder: Generator

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        encoding_dim: int,
        source_encoder_config: DictConfig,
        **kwargs,
    ):
        # base domain encoder init
        base_domain_encoder = super().create(
            seed=seed,
            encoding_dim=encoding_dim,
            source_encoder=None,
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
        new_source_encoder, source_info, source_stats_info = self.source_encoder.update(
            source_random_batch=source_random_batch,
            source_expert_batch=source_expert_batch,
            policy_discriminator=self.policy_discriminator,
            state_discriminator=self.state_discriminator,
        )
        new_encoder = self.replace(
            target_encoder=new_target_encoder,
            source_encoder=new_source_encoder
        )
        info = {**target_info, **source_info}
        stats_info = {**target_stats_info, **source_stats_info}
        return new_encoder, info, stats_info
