from copy import deepcopy

import gymnasium as gym
import jax
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

import wandb
from agents.base_agent import Agent
from agents.imitation_learning.in_domain.gail import GAILDiscriminator
from agents.imitation_learning.utils import get_state_pairs, prepare_buffer
from agents.utils import instantiate_agent
from misc.enot import ENOTGW
from misc.gan.discriminator import Discriminator
from misc.gan.generator import Generator
from utils import SaveLoadMixin, buffer_init, encode_batch, sample_batch_jit
from utils.custom_types import Buffer, BufferState, DataType


class GWILAgent(SaveLoadMixin):
    _save_attrs = (
        "target_learner",
        "source_learner",
        "target_encoder",
        "source_encoder",
        "domain_discriminator",
        "policy_discriminator",
        "gail_discriminator",
        "ot",
        "target_learner_buffer_state",
        "source_learner_buffer_state",
        "source_expert_buffer_state",
    )

    def __init__(
        self,
        seed: int,
        target_env: gym.Env,
        source_env: gym.Env,
        target_learner: Agent,
        source_learner: Agent,
        target_encoder: Generator,
        source_encoder: Generator,
        domain_discriminator: Discriminator,
        policy_discriminator: Discriminator,
        gail_discriminator: GAILDiscriminator,
        ot: ENOTGW,
        buffer: Buffer,
        target_learner_buffer_state: BufferState,
        source_learner_buffer_state: BufferState,
        source_expert_buffer_state: BufferState,
    ):
        self.seed = seed
        self.target_env = target_env
        self.source_env = source_env
        self.target_learner = target_learner
        self.source_learner = source_learner
        self.target_encoder = target_encoder
        self.source_encoder = source_encoder
        self.domain_discriminator = domain_discriminator
        self.policy_discriminator = policy_discriminator
        self.gail_discriminator = gail_discriminator
        self.ot = ot
        self.buffer = buffer
        self.target_learner_buffer_state = target_learner_buffer_state
        self.source_learner_buffer_state = source_learner_buffer_state
        self.source_expert_buffer_state = source_expert_buffer_state

    @classmethod
    def create(
        cls,
        seed: int,
        #
        target_env_config: DictConfig,
        source_env_config: DictConfig,
        #
        target_learner_config: DictConfig,
        source_learner_config: DictConfig,
        #
        encoding_dim: int,
        target_encoder_config: DictConfig,
        source_encoder_config: DictConfig,
        domain_discriminator_config: DictConfig,
        policy_discriminator_config: DictConfig,
        #
        gail_discriminator_config: DictConfig,
        ot_config: DictConfig,
        #
        batch_size: int,
        max_buffer_size: int,
        source_expert_buffer_state_path: str,
    ):
        # Envs init
        target_env = instantiate(target_env_config)
        source_env = instantiate(source_env_config)

        # RL agents init
        target_learner = instantiate_agent(
            target_learner_config,
            seed=seed,
            env=target_env,
            info_key="target_learner",
        )
        source_learner = instantiate_agent(
            source_learner_config,
            seed=seed,
            env=source_env,
            info_key="source_learner",
        )

        # Domain Encoders init
        target_encoder = instantiate(
            target_encoder_config,
            seed=seed,
            input_dim=target_env.observation_space.shape[0],
            output_dim=encoding_dim,
            info_key="target_encoder",
            _recursive_=False,
        )
        source_encoder = instantiate(
            source_encoder_config,
            seed=seed,
            input_dim=source_env.observation_space.shape[0],
            output_dim=encoding_dim,
            info_key="source_encoder",
            _recursive_=False,
        )
        domain_discriminator = instantiate(
            domain_discriminator_config,
            seed=seed,
            input_dim=encoding_dim,
            info_key="domain_discriminator",
            _recursive_=False,
        )
        policy_discriminator = instantiate(
            policy_discriminator_config,
            seed=seed,
            input_dim=encoding_dim*2,
            info_key="policy_discriminator",
            _recursive_=False,
        )

        # GAIL discriminator init
        gail_discriminator = instantiate(
            gail_discriminator_config,
            seed=seed,
            input_dim=encoding_dim*2,
            info_key="gail_discriminator",
            _recursive_=False,
        )

        # OT solver init
        ot = instantiate(
            ot_config,
            seed=seed,
            source_dim=target_env.observation_space.shape[0],
            target_dim=source_env.observation_space.shape[0],
            _recursive_=False,
        )

        # Buffers init
        buffer_config = {
          "_target_": "utils.instantiate_jitted_fbx_buffer",
          "fbx_buffer_config": {
            "_target_": "flashbax.make_item_buffer",
            "max_length": max_buffer_size,
            "min_length": 1,
            "sample_batch_size": batch_size,
            "add_batches": False,
          }
        }

        ## Taret Learner Buffer
        buffer, target_learner_buffer_state = buffer_init(buffer_config, target_env)

        ## Source Learner Buffer
        _, source_learner_buffer_state = buffer_init(buffer_config, source_env)

        ## Source Expert Buffer
        _, source_expert_buffer_state = prepare_buffer(
            buffer_state_path=source_expert_buffer_state_path,
            batch_size=batch_size,
        )

        return cls(
            seed=seed,
            target_env=target_env,
            source_env=source_env,
            target_learner=target_learner,
            source_learner=source_learner,
            target_encoder=target_encoder,
            source_encoder=source_encoder,
            domain_discriminator=domain_discriminator,
            policy_discriminator=policy_discriminator,
            gail_discriminator=gail_discriminator,
            ot=ot,
            buffer=buffer,
            target_learner_buffer_state=target_learner_buffer_state,
            source_learner_buffer_state=source_learner_buffer_state,
            source_expert_buffer_state=source_expert_buffer_state
        )

    def collect_random_buffer(self, n_items: int):
        # target env
        self.target_learner_buffer_state = _collect_random_buffer(
            env=self.target_env,
            buffer=self.buffer,
            state=self.target_learner_buffer_state,
            n_items=n_items,
            seed=self.seed,
            tqdm_desc="TL random buffer collecting",
        )

        # source env
        self.source_learner_buffer_state = _collect_random_buffer(
            env=self.source_env,
            buffer=self.buffer,
            state=self.source_learner_buffer_state,
            n_items=n_items,
            seed=self.seed+1,
            tqdm_desc="SL random buffer collecting",
        )

    def pretrain(self, n_pretrain_iters: int=0):
        for i in tqdm(range(n_pretrain_iters)):
            tl_batch, sl_batch, se_batch = self.sample_batches(seed=self.seed+i)
            tl_batch_encoded, sl_batch_encoded, _ =\
                self._update_domain_encoders(tl_batch, sl_batch, se_batch)
            _ = self._update_ot(tl_batch_encoded, sl_batch_encoded)
            if (i + 1) % self.eval_every == 0:
                self.evaluate(n_episodes=self.n_eval_episodes)
            self.step += 1
        self.evaluate(n_episodes=self.n_eval_episodes)

    def train(
        self,
        random_buffer_size: int,
        n_pretrain_iters: int,
        n_train_iters: int,
        #
        log_every: int,
        save_every: int,
        eval_every: int,
        n_eval_episodes: int,
        wandb_run: "wandb.Run",
    ):
        # store for logging
        self.step = 0
        self.save_every = save_every
        self.log_every = log_every 
        self.eval_every = eval_every 
        self.n_eval_episodes = n_eval_episodes
        self.wandb_run = wandb_run

        # collect random buffers
        self.collect_random_buffer(n_items=random_buffer_size)

        # pretrain
        self.pretrain(n_pretrain_iters=n_pretrain_iters)

        # train
        target_env = deepcopy(self.target_env)
        source_env = deepcopy(self.source_env)
        target_observation, _  = target_env.reset(seed=self.seed)
        source_observation, _  = source_env.reset(seed=self.seed+1)
        for i in tqdm(range(n_train_iters)):
            # update learners buffers
            target_env, target_observation, self.target_learner_buffer_state = self._update_learner_buffer(
                learner=self.target_learner,
                env=target_env,
                observation=target_observation,
                state=self.target_learner_buffer_state,
                seed=self.seed+i,
            )
            source_env, source_observation, self.source_learner_buffer_state = self._update_learner_buffer(
                learner=self.source_learner,
                env=source_env,
                observation=source_observation,
                state=self.source_learner_buffer_state,
                seed=self.seed+i+1,
            )

            # sample batches
            tl_batch, sl_batch, se_batch = self.sample_batches(seed=self.seed+i)

            # update domain encoders
            tl_batch_encoded, sl_batch_encoded, se_batch_encoded =\
                self._update_domain_encoders(tl_batch, sl_batch, se_batch)

            # update optimal transport solver
            tl_batch_encoded_mapped = self._update_ot(tl_batch_encoded, sl_batch_encoded)

            # update gail discriminator
            gail_discr_info, gail_discr_stats_info = self._update_gail_discriminator(
                tl_batch_encoded_mapped, sl_batch_encoded, se_batch_encoded
            )

            # update target learner
            tl_batch["rewards"] = self.gail_discriminator.get_rewards(tl_batch_encoded_mapped)
            tl_info, tl_stats_info = self._update_target_learner(tl_batch)

            # update source learner
            sl_batch["rewards"] = self.gail_discriminator.get_rewards(sl_batch_encoded)
            sl_info, sl_stats_info = self._update_source_learner(sl_batch)

            # logging
            if (i + 1) % self.log_every == 0:
                info = {**gail_discr_info, **tl_info, **sl_info}
                stats_info = {**gail_discr_stats_info, **tl_stats_info, **sl_stats_info}
                for k, v in info.items():
                    wandb_run.log({f"training/{k}": v}, step=self.step)
                for k, v in stats_info.items():
                    wandb_run.log({f"training_stats/{k}": v}, step=self.step)
            if (i + 1) % self.eval_every == 0:
                self.evaluate(n_episodes=self.n_eval_episodes)
            self.step += 1

        self.evaluate(n_episodes=self.n_eval_episodes)


    def _update_learner_buffer(
        self,
        learner: Agent,
        env: gym.Env,
        observation: np.ndarray,
        state: BufferState,
        seed: int,
    ):
        action = learner.sample_actions(key=jax.random.key(seed), observations=observation)
        observation_next, reward, done, truncated, _ = env.step(action)
        state = _update_buffer(
            self.buffer, state, observation, action, reward, done, truncated, observation_next
        )
        if done or truncated:
            observation_next, _ = env.reset(seed=seed)
        return env, observation_next, state

    def _update_target_learner(self, batch: DataType):
        self.target_learner, info, stats_info = self.target_learner.update(batch)
        return info, stats_info

    def _update_source_learner(self, batch: DataType):
        self.source_learner, info, stats_info = self.source_learner.update(batch)
        return info, stats_info

    def _update_gail_discriminator(self, tl_batch_encoded_mapped: DataType, sl_batch_encoded: DataType, se_batch_encoded: DataType):
        batch_size = tl_batch_encoded_mapped["observations"].shape[0]
        self.gail_discriminator, info, stats_info = self.gail_discriminator.update(
            target_expert_batch={
                k: np.concatenate([
                    tl_batch_encoded_mapped[k][:batch_size//2],
                    sl_batch_encoded[k][:batch_size//2]
                ])
                for k in sl_batch_encoded
            },
            source_expert_batch=se_batch_encoded,
        )
        return info, stats_info

    def sample_batches(self, seed: int):
        rng = jax.random.key(seed)
        rng, tl_batch = sample_batch_jit(rng, self.buffer, self.target_learner_buffer_state)
        rng, sl_batch = sample_batch_jit(rng, self.buffer, self.source_learner_buffer_state)
        rng, se_batch = sample_batch_jit(rng, self.buffer, self.source_expert_buffer_state)
        return tl_batch, sl_batch, se_batch

    def _update_domain_encoders(self, tl_batch: DataType, sl_batch: DataType, se_batch: DataType):
        # TODO: identity encoders plugin
        return tl_batch, sl_batch, se_batch

    def _update_ot(self, tl_batch_encoded: DataType, sl_batch_encoded: DataType):
        # TODO: identity mapping plugin
        return tl_batch_encoded

    def evaluate(self, n_episodes: int):
        eval_info = {}

        # process rollouts
        ## target learner
        tl_trajs = _collect_rollouts(
            agent=self.target_learner,
            env=self.target_env,
            num_episodes=n_episodes,
            traj_keys=tuple(self.target_learner_buffer_state.experience.keys()),
            seed=self.seed,
        )
        tl_trajs_encoded = encode_batch(self.target_encoder, tl_trajs)
        tl_trajs_encoded_mapped = encode_batch(self.ot, tl_trajs_encoded)
        tl_trajs["gail_rewards"] = self.gail_discriminator.get_rewards(tl_trajs_encoded_mapped)

        eval_info["TL_TotalRewards"] = np.sum(tl_trajs["rewards"]) / n_episodes
        eval_info["TL_GAILTotalRewards"] = np.sum(tl_trajs["gail_rewards"]) / n_episodes

        ## source learner
        sl_trajs = _collect_rollouts(
            agent=self.source_learner,
            env=self.source_env,
            num_episodes=n_episodes,
            traj_keys=tuple(self.source_learner_buffer_state.experience.keys()),
            seed=self.seed,
        )
        sl_trajs_encoded = encode_batch(self.source_encoder, sl_trajs)
        sl_trajs_encoded_mapped = encode_batch(self.ot, sl_trajs_encoded)
        sl_trajs["gail_rewards"] = self.gail_discriminator.get_rewards(sl_trajs_encoded_mapped)

        eval_info["SL_TotalRewards"] = np.sum(sl_trajs["rewards"]) / n_episodes
        eval_info["SL_GAILTotalRewards"] = np.sum(sl_trajs["gail_rewards"]) / n_episodes

        for k, v in eval_info.items():
            self.wandb_run.log({f"evaluation/{k}": v}, step=self.step)

        return eval_info

def _collect_rollouts(
    agent: Agent,
    env: gym.Env,
    num_episodes: int,
    traj_keys: tuple[str],
    seed: int = 0,
):
    trajs = {traj_key: [] for traj_key in traj_keys}
    for i in range(num_episodes):
        observation, _, done, truncated = *env.reset(seed=seed+i), False, False
        while not (done or truncated):
            action = agent.eval_actions(observation)
            observation_next, reward, done, truncated, _ = env.step(action)

            trajs["observations"].append(observation)
            trajs["actions"].append(action)
            trajs["rewards"].append(reward)
            trajs["dones"].append(done)
            trajs["truncated"].append(done or truncated)
            trajs["observations_next"].append(observation_next)

            observation = observation_next
    for k, v in trajs.items():
        trajs[k] = np.stack(v)
    return trajs

def _collect_random_buffer(
    *,
    env: gym.Env,
    buffer: Buffer,
    state: BufferState,
    n_items: int,
    seed: int,
    tqdm_desc: str="",
):
    observation, _  = env.reset(seed=seed)
    for i in tqdm(range(n_items), desc=tqdm_desc):
        action = env.action_space.sample()
        observation_next, reward, done, truncated, _ = env.step(action)
        state = _update_buffer(
            buffer, state, observation, action, reward, done, truncated, observation_next
        )
        if done or truncated:
            observation_next, _ = env.reset(seed=seed+i)
    return state

def _update_buffer(
    buffer: Buffer,
    state: BufferState,
    observation: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    truncated: np.ndarray,
    observation_next: np.ndarray,
):
    state = buffer.add(
        state,
        {
            "observations": observation,
            "actions": action,
            "rewards": reward,
            "dones": done,
            "truncated": done or truncated,
            "observations_next": observation_next,
        }
    )
    return state
