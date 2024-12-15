import jax
import jax.numpy as jnp
from gan.generator import Generator
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf

from utils.types import DataType

from .base_domain_encoder import BaseDomainEncoder


class CrossDomainEncoder(BaseDomainEncoder):
    expert_encoder: Generator

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        learner_dim: int,
        expert_dim: int,
        encoding_dim: int,
        #
        learner_encoder_config: DictConfig,
        expert_encoder_config: DictConfig,
        state_discriminator_config: DictConfig,
        policy_discriminator_config: DictConfig,
        #
        state_loss_scale: float,
        #
        **kwargs,
    ):
        # base domain encoder init
        base_domain_encoder = super().create(
            seed=seed,
            learner_dim=learner_dim,
            encoding_dim=encoding_dim,
            learner_encoder_config=learner_encoder_config,
            state_discriminator_config=state_discriminator_config,
            policy_discriminator_config=policy_discriminator_config,
            state_loss_scale=state_loss_scale,
            expert_encoder=None,
            **kwargs,
        )

        # expert encoder init
        expert_encoder_config = OmegaConf.to_container(expert_encoder_config)
        expert_encoder_config["loss_config"]["state_loss"] = base_domain_encoder.state_discriminator.state.loss_fn
        expert_encoder_config["loss_config"]["policy_loss"] = base_domain_encoder.policy_discriminator.state.loss_fn

        expert_encoder = instantiate(
            expert_encoder_config,
            seed=seed,
            input_dim=expert_dim,
            output_dim=encoding_dim,
            info_key="expert_encoder",
            _recursive_=False,
        )

        # update _save_attr
        _save_attrs = base_domain_encoder._save_attrs + ("expert_encoder",)

        return base_domain_encoder.replace(
            expert_encoder=expert_encoder,
            _save_attrs=_save_attrs,
        )

    @jax.jit
    def encode_expert_state(self, state: jnp.ndarray):
        return self.expert_encoder(state)

    def _update_encoder(
        self,
        learner_batch: DataType,
        expert_batch: DataType,
    ):
        new_learner_encoder, learner_info, learner_stats_info = self.learner_encoder.update(
            batch=learner_batch,
            policy_discriminator=self.policy_discriminator,
            state_discriminator=self.state_discriminator,
            state_loss_scale=self.state_loss_scale,
        )
        new_expert_encoder, expert_info, expert_stats_info = self.expert_encoder.update(
            batch=expert_batch,
            policy_discriminator=self.policy_discriminator,
            state_discriminator=self.state_discriminator,
            state_loss_scale=self.state_loss_scale,
        )

        new_encoder = self.replace(
            learner_encoder=new_learner_encoder,
            expert_encoder=new_expert_encoder
        )
        info = {**learner_info, **expert_info}
        stats_info = {**learner_stats_info, **expert_stats_info}
        return new_encoder, info, stats_info
