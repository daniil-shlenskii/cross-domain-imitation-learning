import functools
from copy import deepcopy
from typing import Dict, Tuple

import flashbax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from typing_extensions import override

import wandb
from agents.base_agent import Agent
from agents.dida.das import domain_adversarial_sampling
from agents.dida.domain_loss_scale_updaters import \
    IdentityDomainLossScaleUpdater
from agents.dida.update_steps import (update_domain_discriminator_only_jit,
                                      update_gail)
from agents.dida.utils import (encode_observation_jit,
                               get_state_and_policy_tsne_scatterplots)
from agents.gail.gail_discriminator import GAILDiscriminator
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils.types import Buffer, BufferState, DataType, PRNGKey
from utils.utils import (convert_figure_to_array, get_buffer_state_size,
                         instantiate_jitted_fbx_buffer, load_pickle)


class BaseDIDAAgent(Agent):
    agent: Agent
    learner_encoder: Generator
    policy_discriminator: Discriminator
    domain_discriminator: Discriminator
    expert_buffer: Buffer = struct.field(pytree_node=False)
    expert_buffer_state: BufferState = struct.field(pytree_node=False)
    anchor_buffer_state: BufferState = struct.field(pytree_node=False)
    use_das: float = struct.field(pytree_node=False)
    sar_p: float = struct.field(pytree_node=False)
    p_acc_ema: float = struct.field(pytree_node=False)
    p_acc_ema_decay: float = struct.field(pytree_node=False)
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
        n_domain_discriminator_updates: int = 1,
        domain_loss_scale: float = 1.0,
        domain_loss_scale_updater_kwargs: DictConfig = None,
        #
        **kwargs,
    ):  
        # expert buffer init
        expert_buffer_state = load_pickle(expert_buffer_state_path)
        # TODO: truncate state size with the current index value
        expert_buffer = instantiate_jitted_fbx_buffer({
            "_target_": "flashbax.make_item_buffer",
            "sample_batch_size": expert_batch_size,
            "min_length": expert_buffer_state.current_index,
            "max_length": expert_buffer_state.current_index,
            "add_batches": False,
        })

        # anchor buffer init
        anchor_buffer_state = deepcopy(expert_buffer_state)
        buffer_state_size = get_buffer_state_size(anchor_buffer_state)
        perm_idcs = np.random.choice(buffer_state_size)
        anchor_buffer_state.experience["observations_next"] = \
            anchor_buffer_state.experience["observations_next"].at[0, :buffer_state_size].set(
                anchor_buffer_state.experience["observations_next"][0, perm_idcs]
            )
        
        # agent init
        agent = instantiate(
            agent_config,
            seed=seed,
            observation_dim=encoder_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            _recursive_=False,
        )

        # encoders init
        learner_encoder = instantiate(
            learner_encoder_config,
            seed=seed,
            input_dim=observation_dim,
            output_dim=encoder_dim,
            info_key="encoder",
            _recursive_=False,
        )

        # discriminators init
        policy_discriminator = instantiate(
            policy_discriminator_config,
            seed=seed,
            input_dim=encoder_dim * 2,
            info_key="policy_discriminator",
            _recursive_=False,
        )
        domain_discriminator = instantiate(
            domain_discriminator_config,
            seed=seed,
            input_dim=encoder_dim ,
            info_key="domain_discriminator",
            _recursive_=False,
        )

        if not use_das or domain_loss_scale_updater_kwargs is None:
            domain_loss_scale_updater = IdentityDomainLossScaleUpdater()
        else:
            domain_loss_scale_updater = instantiate(domain_loss_scale_updater_kwargs)

        _save_attrs = kwargs.pop(
            "_save_attrs",
            (
                "agent",
                "learner_encoder",
                "policy_discriminator",
                "domain_discriminator",
                "p_acc_ema",
            )
        )

        return cls(
            rng=jax.random.key(seed),
            agent=agent,
            learner_encoder=learner_encoder,
            policy_discriminator=policy_discriminator,
            domain_discriminator=domain_discriminator,
            expert_buffer=expert_buffer,
            expert_buffer_state=expert_buffer_state,
            anchor_buffer_state=anchor_buffer_state,
            use_das=use_das,
            sar_p=sar_p,
            p_acc_ema=p_acc_ema,
            p_acc_ema_decay=p_acc_ema_decay,
            n_domain_discriminator_updates=n_domain_discriminator_updates,
            domain_loss_scale=domain_loss_scale,
            domain_loss_scale_updater=domain_loss_scale_updater,
            _save_attrs=_save_attrs,
            **kwargs,
        )

    @property
    def actor(self):
        return self.agent.actor
    
    def __getattr__(self, item: str):
        if item == "expert_encoder":
            return self.learner_encoder    
        return super().__getattr__(item)
        
    def update(self, batch: DataType):
        update_domain_discriminator_only = (
            self.n_domain_discriminator_updates > 1 and
            (self.domain_discriminator.state.step + 1) % self.n_domain_discriminator_updates != 0
        )

        if update_domain_discriminator_only:
            new_agent, info, stats_info = self._update_domain_discriminator_only(batch=batch)
        else:
            new_domain_loss_scale = self.domain_loss_scale_updater.update(self)
            new_agent, info, stats_info = self._update(
                batch=batch, 
                domain_loss_scale=new_domain_loss_scale
            )
            info["domain_loss_scale"] = new_domain_loss_scale

        return new_agent, info, stats_info

    def _update_domain_discriminator_only(self, batch: DataType):
        (
            new_rng,
            new_domain_discriminator,
            info,
            stats_info
        ) = update_domain_discriminator_only_jit(
            rng=self.rng,
            batch=batch,
            expert_buffer=self.expert_buffer,
            expert_buffer_state=self.expert_buffer_state,
            learner_encoder=self.learner_encoder,
            expert_encoder=self.expert_encoder,
            domain_discriminator=self.domain_discriminator,
        )
        new_agent = self.replace(
            rng=new_rng,
            domain_discriminator=new_domain_discriminator,
        )
        return new_agent, info, stats_info

    def _update(self, batch: DataType, domain_loss_scale: float):
        # update encoders, domain discriminator, prepare batches for gail update
        (
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
        ) = self._update_encoders_and_domain_discrimiantor_with_extra_preparation(
            batch=batch, domain_loss_scale=domain_loss_scale,
        )

        # prepare mixed batch for policy discriminator updapte
        if self.use_das:
            new_rng, mixed_batch, new_p_acc_ema, sar_info = domain_adversarial_sampling(
                rng=new_rng,
                encoded_learner_batch=batch,
                encoded_anchor_batch=anchor_batch,
                learner_domain_logits=learner_domain_logits,
                expert_domain_logits=expert_domain_logits,
                sar_p=self.sar_p,
                p_acc_ema=self.p_acc_ema,
                p_acc_ema_decay=self.p_acc_ema_decay,
            )
        else:
            mixed_batch = batch
            sar_info, new_p_acc_ema = {}, None

        # apply gail
        new_rl_agent, new_policy_disc, gail_info, gail_stats_info = update_gail(
            batch=batch,
            expert_batch=expert_batch,
            mixed_batch=mixed_batch,
            agent=self.agent,
            policy_discriminator=self.policy_discriminator,
        )

        # update agent with new params
        new_agent_params = dict(
            rng=new_rng,
            learner_encoder=new_learner_encoder,
            policy_discriminator=new_policy_disc,
            domain_discriminator=new_domain_disc,
            agent=new_rl_agent,
            p_acc_ema=new_p_acc_ema,  
        )
        if new_expert_encoder is not None:
            new_agent_params["expert_encoder"] = new_expert_encoder
        new_agent = self.replace(**new_agent_params)

        info.update({**gail_info, **sar_info})
        stats_info.update({**gail_stats_info})
        return new_agent, info, stats_info
    
    def evaluate(
        self,
        *,
        seed: int, 
        env: gym.Env,
        num_episodes: int,
        #
        learner_buffer: Buffer,
        learner_buffer_state: BufferState,
        visualize_n_trajectories: int,
        convert_to_wandb_type: bool = True,
    ) -> Dict[str, float]:
        eval_info = super().evaluate(seed=seed, env=env, num_episodes=num_episodes)

        tsne_state_figure, tsne_policy_figure = get_state_and_policy_tsne_scatterplots(
            seed=seed,
            dida_agent=self,
            env=env,
            num_episodes=visualize_n_trajectories,
            expert_buffer_state=self.expert_buffer_state,
            anchor_buffer_state=self.anchor_buffer_state,
        )
        if convert_to_wandb_type:
            tsne_state_figure = wandb.Image(convert_figure_to_array(tsne_state_figure), caption="TSNE plot of state feautures")
            tsne_policy_figure = wandb.Image(convert_figure_to_array(tsne_policy_figure), caption="TSNE plot of policy feautures")

        eval_info["tsne_state_scatter"] = tsne_state_figure
        eval_info["tsne_policy_scatter"] = tsne_policy_figure
        return eval_info

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return encode_observation_jit(self.learner_encoder, observations)

    @override
    def _update_encoders_and_domain_discrimiantor_with_extra_preparation(
        self, batch: DataType, domain_loss_scale: float
    ):
        pass