from typing import Optional

import gymnasium as gym
import jax
import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from agents.imitation_learning.base_imitation_agent import ImitationAgent
from agents.imitation_learning.utils import prepare_buffer
from utils import sample_batch_jit
from utils.custom_types import DataType

from .gwil_enot import GWILENOT


class GWILAgent(ImitationAgent):
    gwil_enot: GWILENOT 
    update_agent_every: int

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        observation_dim: int,
        action_dim: int,
        low: np.ndarray,
        high: np.ndarray,
        #
        agent_config: DictConfig,
        gwil_enot_config: DictConfig,
        #
        source_expert_buffer_state_path: str,
        batch_size: Optional[int],
        sourse_buffer_processor_config: Optional[DictConfig] = None,
        #
        update_agent_every: int = 1,
        **kwargs,
    ):
        # agent init
        agent = instantiate(
            agent_config,
            seed=seed,
            observation_dim=observation_dim,
            action_dim=action_dim,
            low=low,
            high=high,
            _recursive_=False,
        )

        # expert buffer init
        buffer, source_expert_buffer_state = prepare_buffer(
            buffer_state_path=source_expert_buffer_state_path,
            batch_size=batch_size,
            sourse_buffer_processor_config=sourse_buffer_processor_config,
        )

        # gwil_enot init
        gwil_enot = instantiate(
            gwil_enot_config,
            source_dim=observation_dim,
            target_dim=source_expert_buffer_state.experience["observations"].shape[-1],
            _recursive_=False,
        )

        # set attrs to save
        _save_attrs = kwargs.pop(
            "_save_attrs",
            ("agent", "gwil_enot")
        )

        return cls(
            rng=jax.random.key(seed),
            buffer=buffer,
            source_expert_buffer_state=source_expert_buffer_state,
            agent=agent,
            gwil_enot=gwil_enot,
            update_agent_every=update_agent_every,
            _save_attrs = _save_attrs,
            **kwargs,
        )

    @property
    def actor(self):
        return self.agent.actor

    @jax.jit
    def pretrain_update(self, batch: DataType):
        target_expert_batch = batch # renaming

        # sample expert batch
        new_rng, source_expert_batch = sample_batch_jit(
            self.rng, self.buffer, self.source_expert_buffer_state
        )

        # update gwil enot
        new_gwil_enot, gwil_enot_info, gwil_enot_stats_info = self.gwil_enot.update(
            target_expert_batch=target_expert_batch,
            source_expert_batch=source_expert_batch,
        )

        self = self.replace(rng=new_rng, gwil_enot=new_gwil_enot)
        info = {**gwil_enot_info}
        stats_info = {**gwil_enot_stats_info}
        return self, info, stats_info

    @jax.jit
    def update(self, batch: DataType):
        target_expert_batch = batch # renaming

        # sample expert batch
        new_rng, source_expert_batch = sample_batch_jit(
            self.rng, self.buffer, self.source_expert_buffer_state
        )

        # update gwil enot
        new_gwil_enot, gwil_enot_info, gwil_enot_stats_info = self.gwil_enot.update(
            target_expert_batch=target_expert_batch,
            source_expert_batch=source_expert_batch,
        )

        # set gwil rewards
        batch["rewards"] = self.gwil_enot.get_rewards(batch)

        # update agent
        new_agent, agent_info, agent_stats_info = self.agent.update(batch)
        new_agent = jax.lax.cond(
            new_gwil_enot.enot.transport.step % self.update_agent_every == 0,
            lambda: new_agent,
            lambda: self.agent,
        )

        self = self.replace(
            rng=new_rng,
            agent=new_agent,
            gwil_enot=new_gwil_enot,
        )
        info = {**gwil_enot_info, **agent_info}
        stats_info = {**gwil_enot_stats_info, **agent_stats_info}
        return self, info, stats_info

    def evaluate(
        self,
        *,
        seed: int,
        env: gym.Env,
        num_episodes: int,
        #
        return_trajectories: bool = False,
        #
        return_traj_dict: bool = False,
        convert_to_wandb_type: bool = True,
    ):
        eval_info, trajs, traj_dict = super().evaluate(seed=seed, env=env, num_episodes=num_episodes, return_trajectories=True, return_traj_dict=True)

        # enot eval info
        enot_info = self.gwil_enot.evaluate(
            source_pairs=traj_dict["state_pairs"]["TE"],
            target_pairs=traj_dict["state_pairs"]["SE"],
            convert_to_wandb_type=convert_to_wandb_type
        )
        eval_info.update(**enot_info)

        #
        if return_trajectories and return_traj_dict:
            return eval_info, trajs, traj_dict
        if return_traj_dict:
            return eval_info, traj_dict
        if return_trajectories:
            return eval_info, trajs
        return eval_info
