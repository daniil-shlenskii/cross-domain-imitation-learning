import gymnasium as gym
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from agents.base_agent import Agent
from agents.imitation_learning.in_domain.gail import GAILDiscriminator
from agents.imitation_learning.utils import prepare_buffer
from agents.utils import instantiate_agent
from misc.enot import ENOTGW
from misc.gan.discriminator import Discriminator
from misc.gan.generator import Generator
from utils import SaveLoadMixin, buffer_init
from utils.custom_types import Buffer, BufferState


class GWILAgent(SaveLoadMixin):
# class GWILAgent:
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
        )

        # source env
        self.source_learner_buffer_state = _collect_random_buffer(
            env=self.source_env,
            buffer=self.buffer,
            state=self.source_learner_buffer_state,
            n_items=n_items,
            seed=self.seed,
        )

    # def pretrain(self, n_iters: int=0):
    #     for i in tqdm(range(n_iters)):
    #         tl_batch, sl_batch, se_batch = self.sample_batches(seed=self.seed+i)
    #         tl_batch_encoded, sl_batch_encoded, _ =\
    #             self._update_domain_encoders(tl_batch, sl_batch, se_batch)
    #         _ = self._update_ot(tl_batch_encoded, sl_batch_encoded)
    #
    #
    # def train(self, n_iters: int):
    #     for i in tqdm(range(n_iters)):
    #         tl_batch, sl_batch, se_batch = self.sample_batches(seed=self.seed+i)
    #
    #         tl_batch_encoded, sl_batch_encoded, se_batch_encoded =\
    #             self._update_domain_encoders(tl_batch, sl_batch, se_batch)
    #
    #         tl_batch_encoded_mapped = self._update_ot(tl_batch_encoded, sl_batch_encoded)
    #
    #         self._update_gail_discriminator(tl_batch_encoded_mapped, sl_batch_encoded, se_batch_encoded)
    #
    #         tl_batch["rewards"] = self.gail_discriminator.get_rewards(tl_batch_encoded_mapped)
    #         self._update_target_learner(tl_batch)
    #
    #         sl_batch["rewards"] = self.gail_discriminator.get_rewards(sl_batch_encoded)
    #         self._update_source_learner(sl_batch)

    def evaluate(self):
        pass

def _collect_random_buffer(
    *,
    env: gym.Env,
    buffer: Buffer,
    state: BufferState,
    n_items: int,
    seed: int,
):
    observation, _  = env.reset(seed=seed)
    for i in range(n_items):
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
