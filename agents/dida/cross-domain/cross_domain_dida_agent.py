import functools

import jax
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.dida.base_dida_agent import BaseDIDAAgent
from agents.gail.gail_discriminator import GAILDiscriminator
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils.types import Buffer, BufferState, DataType, PRNGKey
from utils.utils import (convert_figure_to_array, get_buffer_state_size,
                         instantiate_jitted_fbx_buffer, load_pickle)


class CrossDomainDIDAAgent(BaseDIDAAgent):
    expert_encoder: Generator

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        encoder_dim: int,
        expert_encoder_config: DictConfig,
        expert_buffer_state_path: str,
        **kwargs,
    ):  
        expert_buffer_state = load_pickle(expert_buffer_state_path)
        expert_observation_dim = expert_buffer_state.experience["observations"].shape[-1]
        expert_encoder = instantiate(
            expert_encoder_config,
            seed=seed,
            input_dim=expert_observation_dim,
            output_dim=encoder_dim,
            info_key="expert_encoder",
            _recursive_=False,
        )
        return super().create(
            expert_encoder=expert_encoder,
            _save_attrs=(
                "agent",
                "learner_encoder",
                "expert_encoder",
                "policy_discriminator",
                "domain_discriminator",
                "p_acc_ema",
            ),
            #
            seed=seed,
            encoder_dim=encoder_dim,
            expert_buffer_state_path=expert_buffer_state_path,
            **kwargs,
        )
    
    def _update_encoders_and_domain_discrimiantor_with_extra_preparation(
        self, batch: DataType, domain_loss_scale: float
    ):
        return _update_encoders_and_domain_discrimiantor_with_extra_preparation_jit(
            rng=self.rng,
            batch=batch,
            expert_buffer=self.expert_buffer,
            expert_buffer_state=self.expert_buffer_state,
            anchor_buffer_state=self.anchor_buffer_state,
            learner_encoder=self.learner_encoder,
            expert_encoder=self.expert_encoder,
            policy_discriminator=self.policy_discriminator,
            domain_discriminator=self.domain_discriminator,
            domain_loss_scale=domain_loss_scale,
        )

@functools.partial(jax.jit, static_argnames="expert_buffer")
def _update_encoders_and_domain_discrimiantor_with_extra_preparation_jit(
    *,
    rng: PRNGKey,
    batch: DataType,
    expert_buffer: Buffer,
    expert_buffer_state: BufferState,
    anchor_buffer_state: BufferState,
    learner_encoder: Generator,
    expert_encoder: Generator,
    policy_discriminator: GAILDiscriminator,
    domain_discriminator: Discriminator,
    domain_loss_scale: float,
):
    # prepare batches
    new_rng, expert_key, anchor_key = jax.random.split(rng, 3)
    expert_batch = expert_buffer.sample(expert_buffer_state, expert_key).experience
    anchor_batch = expert_buffer.sample(anchor_buffer_state, anchor_key).experience

    # update encoders
    new_learner_encoder, learner_encoder_info, learner_encoder_stats_info = learner_encoder.update(
        batch=batch,
        policy_discriminator=policy_discriminator,
        domain_discriminator=domain_discriminator,
        domain_loss_scale=domain_loss_scale,
    )
    new_expert_encoder, expert_encoder_info, expert_encoder_stats_info = expert_encoder.update(
        batch=expert_batch,
        policy_discriminator=policy_discriminator,
        domain_discriminator=domain_discriminator,
        domain_loss_scale=domain_loss_scale,
    )
    encoded_batch = learner_encoder_info.pop("encoded_batch")
    encoded_expert_batch = expert_encoder_info.pop("encoded_batch")

    # update domain discriminator
    new_domain_disc, domain_disc_info, domain_disc_stats_info = domain_discriminator.update(
        real_batch=encoded_batch["observations"],
        fake_batch=encoded_expert_batch["observations"],
        return_logits=True,
    )
    learner_domain_logits = domain_disc_info.pop("real_logits")
    expert_domain_logits = domain_disc_info.pop("fake_logits")

    info = {**learner_encoder_info, **expert_encoder_info, **domain_disc_info}
    stats_info = {**learner_encoder_stats_info, **expert_encoder_stats_info, **domain_disc_stats_info}

    # encode anchor batch for das method
    anchor_batch["observations"] = expert_encoder(anchor_batch["observations"])
    anchor_batch["observations_next"] = expert_encoder(anchor_batch["observations_next"])

    return (
        new_rng,
        new_learner_encoder,
        new_expert_encoder,
        new_domain_disc,
        batch,
        expert_batch,
        anchor_batch,
        learner_domain_logits,
        expert_domain_logits,
        info,
        stats_info,
    )