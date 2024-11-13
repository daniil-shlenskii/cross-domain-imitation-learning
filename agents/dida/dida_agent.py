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

import wandb
from agents import GAILAgent
from agents.base_agent import Agent
from agents.dida.das import domain_adversarial_sampling
from agents.dida.sar import self_adaptive_rate
from agents.dida.utils import (encode_observation, get_tsne_embeddings_scatter,
                               process_policy_discriminator_input)
from agents.gail.gail_discriminator import GAILDiscriminator
from gan.discriminator import Discriminator
from gan.generator import Generator
from utils.types import Buffer, BufferState, DataType, PRNGKey
from utils.utils import (convert_figure_to_array, get_buffer_state_size,
                         load_pickle)


class DIDAAgent(GAILAgent):
    anchor_buffer_state: BufferState = struct.field(init=False)
    learner_encoder: Generator
    expert_encoder: Generator
    domain_discriminator: Discriminator
    use_das: float = struct.field(pytree_node=False)
    sar_p: float = struct.field(pytree_node=False)
    p_acc_ema: float = struct.field(pytree_node=False)
    p_acc_ema_decay: float = struct.field(pytree_node=False)
    n_domain_discriminator_updates: int = struct.field(pytree_node=False)
    encoders_domain_discriminator_loss_scale: int = struct.field(pytree_node=False)

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
        encoders_dim: int,
        #
        expert_batch_size: int,
        expert_buffer_state_path: str,
        #
        agent_config: DictConfig,
        discriminator_config: DictConfig,
        #
        learner_encoder_config: DictConfig,
        expert_encoder_config: DictConfig,
        domain_discriminator_config: DictConfig,
        #
        use_das: bool = True,
        sar_p: float = 0.66,
        p_acc_ema: float = 0.85,
        p_acc_ema_decay: float = 0.999,
        #
        n_domain_discriminator_updates: int = 1,
        encoders_domain_discriminator_loss_scale: float = 0.2,
    ):  
        discriminator_config["info_key"] = "policy_discriminator"

        learner_encoder_input_sample = jnp.ones(observation_dim)
        learner_encoder = instantiate(
            learner_encoder_config,
            seed=seed,
            input_sample=learner_encoder_input_sample,
            output_dim=encoders_dim,
            info_key="learner_encoder",
            _recursive_=False,
        )

        expert_buffer_state = load_pickle(expert_buffer_state_path)
        expert_observation_dim = expert_buffer_state.experience["observations"].shape[-1]
        expert_encoder_input_sample = jnp.ones(expert_observation_dim)
        expert_encoder = instantiate(
            expert_encoder_config,
            seed=seed,
            input_sample=expert_encoder_input_sample,
            output_dim=encoders_dim,
            info_key="expert_encoder",
            _recursive_=False,
        )

        domain_discriminator_input_sample = np.ones(encoders_dim)
        domain_discriminator = instantiate(
            domain_discriminator_config,
            seed=seed,
            input_sample=domain_discriminator_input_sample,
            info_key="domain_discriminator",
            _recursive_=False,
        )

        return super().create(
            seed=seed,
            observation_dim=encoders_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            expert_batch_size=expert_batch_size,
            expert_buffer_state_path=expert_buffer_state_path,
            agent_config=agent_config,
            discriminator_config=discriminator_config,
            #
            learner_encoder=learner_encoder,
            expert_encoder=expert_encoder,
            domain_discriminator=domain_discriminator,
            use_das=use_das,
            sar_p=sar_p,
            n_domain_discriminator_updates=n_domain_discriminator_updates,
            encoders_domain_discriminator_loss_scale=encoders_domain_discriminator_loss_scale,
            p_acc_ema_decay=p_acc_ema_decay,
            p_acc_ema=p_acc_ema,
            #
            _save_attrs = (
                "agent",
                "discriminator",
                "learner_encoder",
                "expert_encoder",
                "domain_discriminator",
                "p_acc_ema",
            ),
        )
    
    def __post_init__(self):
        # prepare anchor replay buffer
        anchor_buffer_state = deepcopy(self.expert_buffer_state)
        buffer_state_size = get_buffer_state_size(anchor_buffer_state)
        perm_idcs = np.random.choice(buffer_state_size)
        anchor_buffer_state.experience["observations_next"] = \
            anchor_buffer_state.experience["observations_next"].at[0, :buffer_state_size].set(
                anchor_buffer_state.experience["observations_next"][0, perm_idcs]
            )
        object.__setattr__(self, 'anchor_buffer_state', anchor_buffer_state)

    def update(self, batch: DataType):
        update_domain_discriminator_only = (
            self.n_domain_discriminator_updates > 1 and
            (self.domain_discriminator.state.step + 1) % self.n_domain_discriminator_updates != 0
        )

        if update_domain_discriminator_only:
            (
                new_rng,
                new_domain_discriminator,
                info,
                stats_info
            ) = _update_domain_discriminator_jit(
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
        else:
            (
                new_rng,
                new_learner_encoder,
                new_expert_encoder,
                new_policy_discriminator,
                new_domain_discriminator,
                new_rl_agent,
                new_p_acc_ema,
                info,
                stats_info
            ) = _update_jit(
                rng=self.rng,
                batch=batch,
                expert_buffer=self.expert_buffer,
                expert_buffer_state=self.expert_buffer_state,
                anchor_buffer_state=self.anchor_buffer_state,
                learner_encoder=self.learner_encoder,
                expert_encoder=self.expert_encoder,
                policy_discriminator=self.discriminator,
                domain_discriminator=self.domain_discriminator,
                agent=self.agent,
                use_das=self.use_das,
                sar_p=self.sar_p,
                p_acc_ema=self.p_acc_ema,
                p_acc_ema_decay=self.p_acc_ema_decay,
                encoders_domain_discriminator_loss_scale=self.encoders_domain_discriminator_loss_scale
            )
            new_agent = self.replace(
                rng=new_rng,
                learner_encoder=new_learner_encoder,
                expert_encoder=new_expert_encoder,
                discriminator=new_policy_discriminator,
                domain_discriminator=new_domain_discriminator,
                agent=new_rl_agent,
                p_acc_ema=new_p_acc_ema,
            )
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
        n_samples_per_buffer: int,
        convert_to_wandb_type: bool = True,
    ) -> Dict[str, float]:
        eval_info = super().evaluate(seed=seed, env=env, num_episodes=num_episodes)

        tsne_state_figure, tsne_behavior_figure = get_tsne_embeddings_scatter(
            seed=seed,
            learner_buffer=learner_buffer,
            expert_buffer=self.expert_buffer,
            learner_buffer_state=learner_buffer_state,
            expert_buffer_state=self.expert_buffer_state,
            anchor_buffer_state=self.anchor_buffer_state,
            learner_encoder=self.learner_encoder,
            expert_encoder=self.expert_encoder,
            n_samples_per_buffer=n_samples_per_buffer,
        )
        if convert_to_wandb_type:
            tsne_state_figure = wandb.Image(convert_figure_to_array(tsne_state_figure), caption="TSNE plot of state feautures")
            tsne_behavior_figure = wandb.Image(convert_figure_to_array(tsne_behavior_figure), caption="TSNE plot of behavior feautures")
        eval_info["tsne_state_scatter"] = tsne_state_figure
        eval_info["tsne_behvaior_scatter"] = tsne_behavior_figure
        return eval_info

    def _preprocess_observations(self, observations: np.ndarray) -> np.ndarray:
        return encode_observation(self.learner_encoder, observations)

def _update_jit(
    *,
    rng: PRNGKey,
    batch: DataType,
    expert_buffer: Buffer,
    expert_buffer_state: BufferState,
    anchor_buffer_state: BufferState,
    #
    learner_encoder: Generator,
    expert_encoder: Generator,
    policy_discriminator: Discriminator,
    domain_discriminator: Discriminator,
    agent: Agent,
    #
    use_das: bool,
    sar_p: float,
    p_acc_ema: float,
    p_acc_ema_decay: float,
    encoders_domain_discriminator_loss_scale,
):
    new_rng, expert_batch, anchor_batch = _prepare_batches_step_jit(rng, expert_buffer, expert_buffer_state, anchor_buffer_state)

    # UPDATE encoders
    (
        new_learner_encoder,
        new_expert_encoder,
        encoded_learner_domain_batch,
        encoded_expert_domain_batch,
        encoded_learner_policy_batch,
        encoded_expert_policy_batch,
        encoders_info,
        encoders_stats_info,
    ) = _update_encoders_step_jit(
        learner_batch=batch,
        expert_batch=expert_batch,
        learner_encoder=learner_encoder,
        expert_encoder=expert_encoder,
        policy_discriminator=policy_discriminator,
        domain_discriminator=domain_discriminator,
        encoders_domain_discriminator_loss_scale=encoders_domain_discriminator_loss_scale
    )

    # UPDATE domain discriminator
    new_domain_disc, domain_disc_info, domain_disc_stats_info = _update_domain_discriminator_step_jit(
        encoded_learner_domain_batch=encoded_learner_domain_batch,
        encoded_expert_domain_batch=encoded_expert_domain_batch,
        domain_discriminator=domain_discriminator
    )

    # UPDATE agent and policy discriminator with gail
    batch, expert_batch, anchor_batch = _prepare_gail_batches_jit(
        batch=batch,
        expert_batch=expert_batch,
        anchor_batch=anchor_batch,
        encoded_learner_policy_batch=encoded_learner_policy_batch,
        encoded_expert_policy_batch=encoded_expert_policy_batch,
        expert_encoder=expert_encoder,
    )

    if use_das:
        alpha, new_p_acc_ema, sar_info = self_adaptive_rate(
            domain_discriminator=domain_discriminator,
            learner_batch=batch,
            expert_batch=expert_batch,
            p=sar_p,
            p_acc_ema=p_acc_ema,
            p_acc_ema_decay=p_acc_ema_decay,
        )
        new_rng, mixed_batch = domain_adversarial_sampling(
            rng=new_rng,
            embedded_learner_batch=batch,
            embedded_anchor_batch=anchor_batch,
            domain_discriminator=domain_discriminator,
            alpha=alpha
        )
    else:
        batch_size = batch["observations"].shape[0]
        mixed_batch = jax.tree.map(
            lambda x, y: jnp.concatenate([x[:batch_size//2], y[:batch_size//2]], axis=0),
            batch,
            anchor_batch
        )
        sar_info, new_p_acc_ema = {}, None

    # apply gail
    new_agent, new_policy_disc, gail_info, gail_stats_info = _update_gail_step_jit(
        batch=batch,
        expert_batch=expert_batch,
        mixed_batch=mixed_batch,
        agent=agent,
        discriminator=policy_discriminator,
    )

    info = {**encoders_info, **domain_disc_info, **gail_info, **sar_info}
    stats_info = {**encoders_stats_info, **domain_disc_stats_info, **gail_stats_info}
    return (
        new_rng,
        new_learner_encoder,
        new_expert_encoder,
        new_policy_disc,
        new_domain_disc,
        new_agent,
        new_p_acc_ema,
        info,
        stats_info
    )

@functools.partial(jax.jit, static_argnames="expert_buffer")
def _prepare_batches_step_jit(
    rng: PRNGKey,
    expert_buffer: Buffer,
    expert_buffer_state: BufferState,
    anchor_buffer_state: BufferState
):
    new_rng, expert_key, anchor_key = jax.random.split(rng, 3)
    expert_batch = expert_buffer.sample(expert_buffer_state, expert_key).experience
    anchor_batch = expert_buffer.sample(anchor_buffer_state, anchor_key).experience
    return new_rng, expert_batch, anchor_batch

@jax.jit
def _update_encoders_step_jit(
    learner_batch: DataType,
    expert_batch: DataType,
    learner_encoder: Generator,
    expert_encoder: Generator,
    policy_discriminator: Discriminator,
    domain_discriminator: Discriminator,
    encoders_domain_discriminator_loss_scale: float,
):
    # update encoders with policy discriminator
    learner_policy_batch = jnp.concatenate([learner_batch["observations"], learner_batch["observations_next"]], axis=0)
    expert_policy_batch = jnp.concatenate([expert_batch["observations"], expert_batch["observations_next"]], axis=0)
    
    new_learner_encoder, learner_encoder_info, learner_encoder_stats_info = learner_encoder.update(
        hold_grad=True,
        grad_scale=encoders_domain_discriminator_loss_scale,
        batch=learner_policy_batch,
        discriminator=policy_discriminator,
        process_discriminator_input=process_policy_discriminator_input
    )
    new_expert_encoder, expert_encoder_info, expert_encoder_stats_info = expert_encoder.update(
        hold_grad=True,
        grad_scale=encoders_domain_discriminator_loss_scale,
        batch=expert_policy_batch,
        discriminator=policy_discriminator,
        process_discriminator_input=process_policy_discriminator_input
    )

    # store encoded batch for policy discriminator update
    encoded_learner_policy_batch = learner_encoder_info.pop("generations")
    encoded_expert_policy_batch = expert_encoder_info.pop("generations")

    info = {"upd_with_policy_disc": {**learner_encoder_info, **expert_encoder_info}}
    stats_info = {"upd_with_policy_disc": {**learner_encoder_stats_info, **expert_encoder_stats_info}}

    # update encoders with domain discriminator
    learner_domain_batch = learner_batch["observations"]
    expert_domain_batch = expert_batch["observations"]

    new_learner_encoder, learner_encoder_info, learner_encoder_stats_info = new_learner_encoder.update(
        batch=learner_domain_batch, discriminator=domain_discriminator
    )
    new_expert_encoder, expert_encoder_info, expert_encoder_stats_info = new_expert_encoder.update(
        batch=expert_domain_batch, discriminator=domain_discriminator
    )

    # store encoded batch for domain discriminator update
    encoded_learner_domain_batch = learner_encoder_info.pop("generations")
    encoded_expert_domain_batch = expert_encoder_info.pop("generations")

    info["upd_with_domain_disc"] = {**learner_encoder_info, **expert_encoder_info}
    stats_info["upd_with_domain_disc"] = {**learner_encoder_stats_info, **expert_encoder_stats_info}

    return (
        new_learner_encoder,
        new_expert_encoder,
        encoded_learner_domain_batch,
        encoded_expert_domain_batch,
        encoded_learner_policy_batch,
        encoded_expert_policy_batch,
        info,
        stats_info,
    )

@jax.jit
def _update_domain_discriminator_step_jit(
    *,
    encoded_learner_domain_batch: jnp.ndarray,
    encoded_expert_domain_batch: jnp.ndarray,
    domain_discriminator: Discriminator,
):
    new_domain_disc, domain_disc_info, domain_disc_stats_info = domain_discriminator.update(
        real_batch=encoded_learner_domain_batch,
        fake_batch=encoded_expert_domain_batch
    )
    return new_domain_disc, domain_disc_info, domain_disc_stats_info

@functools.partial(jax.jit, static_argnames="expert_buffer")
def _update_domain_discriminator_jit(
    *,
    rng: PRNGKey,
    batch: DataType,
    expert_buffer: Buffer,
    expert_buffer_state: BufferState,
    learner_encoder: Generator,
    expert_encoder: Generator,
    domain_discriminator: Discriminator
):
    new_rng, key = jax.random.split(rng, 2)
    expert_batch = expert_buffer.sample(expert_buffer_state, key).experience

    encoded_learner_domain_batch = learner_encoder(batch["observations"])
    encoded_expert_domain_batch = expert_encoder(expert_batch["observations"])

    new_domain_disc, domain_disc_info, domain_disc_stats_info = domain_discriminator.update(
        real_batch=encoded_learner_domain_batch,
        fake_batch=encoded_expert_domain_batch
    )
    return new_rng, new_domain_disc, domain_disc_info, domain_disc_stats_info

@jax.jit
def _prepare_gail_batches_jit(
    batch: DataType,
    expert_batch: DataType,
    anchor_batch: DataType,
    encoded_learner_policy_batch: jnp.ndarray,
    encoded_expert_policy_batch: jnp.ndarray,
    expert_encoder: Generator,
):
    encoders_dim = encoded_learner_policy_batch.shape[-1] // 2
    batch["observations"] = encoded_learner_policy_batch[:, :encoders_dim]
    batch["observations_next"] = encoded_learner_policy_batch[:, encoders_dim:]
    expert_batch["observations"] = encoded_expert_policy_batch[:, :encoders_dim]
    expert_batch["observations_next"] = encoded_expert_policy_batch[:, encoders_dim:]
    anchor_batch["observations"] = expert_encoder(anchor_batch["observations"])
    anchor_batch["observations_next"] = expert_encoder(anchor_batch["observations_next"])

    return batch, expert_batch, anchor_batch

@jax.jit
def _update_gail_step_jit(
    *,
    batch: DataType,
    expert_batch: DataType,
    mixed_batch: DataType,
    #
    agent: Agent,
    discriminator: GAILDiscriminator
):
    policy_batch = jnp.concatenate([batch["observations"], batch["observations_next"]], axis=1)
    expert_policy_batch = jnp.concatenate([expert_batch["observations"], expert_batch["observations_next"]], axis=1)
    mixed_policy_batch = jnp.concatenate([mixed_batch["observations"], mixed_batch["observations_next"]], axis=1)

    # update agent
    batch["reward"] = discriminator.get_rewards(policy_batch)
    new_agent, agent_info, agent_stats_info = agent.update(batch)

    # update discriminator
    new_disc, disc_info, disc_stats_info = discriminator.update(learner_batch=mixed_policy_batch, expert_batch=expert_policy_batch)

    info = {**agent_info, **disc_info}
    stats_info = {**agent_stats_info, **disc_stats_info}
    return new_agent, new_disc, info, stats_info