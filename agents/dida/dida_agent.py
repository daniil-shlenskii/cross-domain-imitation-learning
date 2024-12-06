from copy import deepcopy

import gymnasium as gym
import jax
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from agents.gail.gail_agent import GAILAgent
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils import apply_model_jit, get_buffer_state_size, sample_batch
from utils.types import BufferState, DataType

from .das import DomainAdversarialSampling
from .domain_loss_scale_updaters import IdentityDomainLossScaleUpdater


class DIDAAgent(GAILAgent):
    learner_encoder: Generator
    domain_discriminator: Discriminator
    anchor_buffer_state: BufferState = struct.field(pytree_node=False)
    das: float = struct.field(pytree_node=False)
    n_domain_discriminator_updates: int = struct.field(pytree_node=False)
    domain_loss_scale: float = struct.field(pytree_node=False)
    domain_loss_scale_updater: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        observation_dim: gym.Space,
        action_dim: gym.Space,
        low: np.ndarray[float],
        high: np.ndarray[float],
        #
        encoder_dim: int,
        #
        expert_batch_size: int,
        expert_buffer_state_path: str,
        #
        agent_config: DictConfig,
        learner_encoder_config: DictConfig,
        policy_discriminator_config: DictConfig,
        domain_discriminator_config: DictConfig,
        #
        use_das: bool = True,
        sar_p: float = 0.5,
        p_acc_ema: float = 0.9,
        p_acc_ema_decay: float = 0.999,
        #
        n_policy_discriminator_updates: int = 1,
        #
        n_domain_discriminator_updates: int = 1,
        domain_loss_scale: float = 1.0,
        domain_loss_scale_updater_config: DictConfig = None,
        #
        expert_buffer_state_preprocessing_config: DictConfig = None, # TODO: expert batch preprocessing
        **kwargs,
    ):
        # encoders init
        policy_loss = instantiate(policy_discriminator_config["loss_config"])
        domain_loss = instantiate(domain_discriminator_config["loss_config"])
        learner_encoder_config = OmegaConf.to_container(learner_encoder_config)
        learner_encoder_config["loss_config"]["policy_loss"] = policy_loss
        learner_encoder_config["loss_config"]["domain_loss"] = domain_loss

        learner_encoder = instantiate(
            learner_encoder_config,
            seed=seed,
            input_dim=observation_dim,
            output_dim=encoder_dim,
            info_key="encoder",
            _recursive_=False,
        )

        # domain discriminators init
        domain_discriminator = instantiate(
            domain_discriminator_config,
            seed=seed,
            input_dim=encoder_dim ,
            info_key="domain_discriminator",
            _recursive_=False,
        )

        # DAS init
        das = None
        if use_das:
            das = DomainAdversarialSampling(
                sar_p=sar_p,
                p_acc_ema=p_acc_ema,
                p_acc_ema_decay=p_acc_ema_decay,
            )

        # domain loss updater init
        if not use_das or domain_loss_scale_updater_config is None:
            domain_loss_scale_updater = IdentityDomainLossScaleUpdater()
        else:
            domain_loss_scale_updater = instantiate(domain_loss_scale_updater_config)

        _save_attrs = kwargs.pop(
            "_save_attrs",
            (
                "agent",
                "learner_encoder",
                "policy_discriminator",
                "domain_discriminator",
                "das",
            )
        )

        dida_agent = super().create(
            seed=seed,
            observation_dim=observation_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            expert_batch_size=expert_batch_size,
            expert_buffer_state_path=expert_buffer_state_path,
            agent_config=agent_config,
            policy_discriminator_config=policy_discriminator_config,
            n_policy_discriminator_updates=n_policy_discriminator_updates,
            #
            anchor_buffer_state=None,
            learner_encoder=learner_encoder,
            das=das,
            domain_discriminator=domain_discriminator,
            n_domain_discriminator_updates=n_domain_discriminator_updates,
            domain_loss_scale=domain_loss_scale,
            domain_loss_scale_updater=domain_loss_scale_updater,
            _save_attrs=_save_attrs,
            **kwargs,
        )

        # anchor buffer init
        buffer_state_size = get_buffer_state_size(dida_agent.expert_buffer_state)
        anchor_buffer_state = deepcopy(dida_agent.expert_buffer_state)
        perm_idcs = np.random.choice(buffer_state_size)
        anchor_buffer_state.experience["observations_next"][0] = \
                anchor_buffer_state.experience["observations_next"][0, perm_idcs]

        dida_agent = dida_agent.replace(anchor_buffer_state=anchor_buffer_state)
        return dida_agent

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return apply_model_jit(self.learner_encoder, observations)

    def __getattr__(self, item: str):
        if item == "expert_encoder":
            return self.learner_encoder
        return super().__getattr__(item)

    def update(self, batch: DataType):
        update_gail_agent = bool(
                (self.domain_discriminator.state.step + 1) % self.n_domain_discriminator_updates == 0
        )
        update_agent = update_gail_agent and bool(
                (self.policy_discriminator.state.step + 1) % self.n_policy_discriminator_updates == 0
        )
        new_dida_agent, info, stats_info = self._update(
            batch, update_gail_agent=update_gail_agent, update_agent=update_agent
        )
        return new_dida_agent, info, stats_info

    def _update(self, batch: DataType, update_gail_agent: bool, update_agent: bool):
        # sample expert batch
        new_rng, expert_batch = sample_batch(
            self.rng, self.expert_buffer, self.expert_buffer_state
        )

        # domain discriminator update only
        if not update_gail_agent:
            # encode observations
            batch["observations"] = apply_model_jit(self.learner_encoder, batch["observations"])
            expert_batch["observations"] = apply_model_jit(self.expert_encoder, expert_batch["observations"])

            # update domain discriminator
            new_domain_disc, domain_disc_info, domain_disc_stats_info = self.domain_discriminator.update(
                real_batch=expert_batch["observations"],
                fake_batch=batch["observations"],
            )

            # update gail agent
            new_dida_agent = self.replace(
                rng=new_rng,
                domain_discriminator=new_domain_disc
            )
            return new_dida_agent, domain_disc_info, domain_disc_stats_info

        # update encoders and domain discriminator
        new_domain_loss_scale = self.domain_loss_scale_updater.update(self)
        (
            new_dida_agent,
            batch,
            expert_batch,
            learner_domain_logits,
            expert_domain_logits,
            info,
            stats_info,
        ) = self._update_encoders_and_domain_discrimiantor(
            batch=batch,
            expert_batch=expert_batch,
            domain_loss_scale=new_domain_loss_scale,
        )

        # prepare mixed batch for policy discriminator update
        if new_dida_agent.das is not None:
            # sample anchor batch
            new_rng, anchor_batch = sample_batch(
                new_rng, self.expert_buffer, self.anchor_buffer_state
            )

            # encode anchor batch
            anchor_batch["observations"] = apply_model_jit(self.expert_encoder, anchor_batch["observations"])
            anchor_batch["observations_next"] = apply_model_jit(self.expert_encoder, anchor_batch["observations_next"])

            # mix batches
            mixed_batch, sar_info = new_dida_agent.das.mix_batches(
                learner_batch=batch,
                anchor_batch=anchor_batch,
                learner_domain_logits=learner_domain_logits,
                expert_domain_logits=expert_domain_logits,
            )
            info.update(sar_info)
        else:
            mixed_batch = batch

        # update agent and policy discriminator
        new_dida_agent, gail_info, gail_stats_info = new_dida_agent.update_gail(
            batch=batch,
            expert_batch=expert_batch,
            policy_discriminator_learner_batch=mixed_batch,
            update_agent=update_agent,
        )

        # update dida agent
        new_dida_agent = new_dida_agent.replace(rng=new_rng, domain_loss_scale=new_domain_loss_scale)
        new_dida_agent = new_dida_agent.replace(rng=new_rng)
        info.update(gail_info)
        stats_info.update(gail_stats_info)

        return new_dida_agent, info, stats_info

    @jax.jit
    def _update_encoders_and_domain_discrimiantor(
        self, batch: DataType, expert_batch: DataType, domain_loss_scale: float
    ):
        # update encoders
        new_encoder, encoder_info, encoder_stats_info = self.learner_encoder.update(
            batch=batch,
            expert_batch=expert_batch,
            policy_discriminator=self.policy_discriminator,
            domain_discriminator=self.domain_discriminator,
            domain_loss_scale=domain_loss_scale,
        )
        batch = encoder_info.pop("learner_encoded_batch")
        expert_batch = encoder_info.pop("expert_encoded_batch")

        # update domain discriminator
        new_domain_disc, domain_disc_info, domain_disc_stats_info = self.domain_discriminator.update(
            real_batch=expert_batch["observations"],
            fake_batch=batch["observations"],
            return_logits=True,
        )
        expert_domain_logits = domain_disc_info.pop("real_logits")
        learner_domain_logits = domain_disc_info.pop("fake_logits")

        # update dida agent
        new_dida_agent = self.replace(
            learner_encoder=new_encoder,
            domain_discriminator=new_domain_disc,
        )
        info = {**encoder_info, **domain_disc_info}
        stats_info = {**encoder_stats_info, **domain_disc_stats_info}

        return (
            new_dida_agent,
            batch,
            expert_batch,
            learner_domain_logits,
            expert_domain_logits,
            info,
            stats_info,
        )
