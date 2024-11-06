import functools
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training.train_state import TrainState as FlaxTrainState
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from ott.geometry import costs
from ott.neural.networks.potentials import (BasePotential,
                                            PotentialGradientFn_t,
                                            PotentialTrainState,
                                            PotentialValueFn_t)

from agents.base_agent import Agent
from enot.enot import ExpectileNeuralDual
from nn.train_state import TrainState
from utils.types import DataType, Params, PRNGKey
from utils.utils import instantiate_optimizer, load_pickle, save_pickle


class ENOT:
    _save_ot_solver_attrs: Tuple[str] = (
        "state_f",
        "state_g",
    )

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        data_dim: int,
        #
        f_potential_module_config: DictConfig,
        g_potential_module_config: DictConfig,
        #
        f_potential_optimizer_config: DictConfig,
        g_potential_optimizer_config: DictConfig,
        #
        cost_fn_config: DictConfig,
        expectile: float,
        expectile_loss_coef: float,
        target_weight: float = 1.,
        is_bidirectional: bool = True,
        use_dot_product: bool = False,
        #
        info_key: str = "enot",
    ) -> "ENOT":
        f_potential_module = instantiate(f_potential_module_config)
        g_potential_module = instantiate(g_potential_module_config)
        
        f_potential_optimizer = instantiate_optimizer(f_potential_optimizer_config)
        g_potential_optimizer = instantiate_optimizer(g_potential_optimizer_config)

        cost_fn = instantiate(cost_fn_config)

        return cls(
            seed=seed,
            data_dim=data_dim,
            f_potential_module=f_potential_module,
            g_potential_module=g_potential_module,
            f_potential_optimizer=f_potential_optimizer,
            g_potential_optimizer=g_potential_optimizer,
            cost_fn=cost_fn,
            expectile=expectile,
            expectile_loss_coef=expectile_loss_coef,
            target_weight=target_weight,
            is_bidirectional=is_bidirectional,
            use_dot_product=use_dot_product,
            info_key=info_key,
        )

    def __init__(
        self,
        *,
        seed: int,
        data_dim: int,
        #
        f_potential_module: nn.Module,
        g_potential_module: nn.Module,
        f_potential_optimizer: optax.GradientTransformation,
        g_potential_optimizer: optax.GradientTransformation,
        #
        cost_fn: costs.CostFn,
        expectile: float,
        expectile_loss_coef: float,
        target_weight: float,
        is_bidirectional: bool = True,
        use_dot_product: bool = False,
        #
        info_key: str = "enot",
    ):
        _plug = None
        self.ot_solver = ExpectileNeuralDual(
            dim_data=data_dim,
            neural_f=f_potential_module,
            neural_g=g_potential_module,
            optimizer_f=f_potential_optimizer,
            optimizer_g=g_potential_optimizer,
            cost_fn=cost_fn,
            is_bidirectional=is_bidirectional,
            use_dot_product=use_dot_product,
            expectile=expectile,
            expectile_loss_coef=expectile_loss_coef,
            target_weight=target_weight,
            # 
            rng=jax.random.key(seed),
            num_train_iters=_plug,
            valid_freq=_plug,
            log_freq=_plug,
        )
        self.info_key = info_key

        self._forward_update_turn = False

    def __call__(self, source: jnp.ndarray) -> jnp.ndarray:
        learned_potentials = self.to_dual_potentials()
        return learned_potentials.transport(source)

    def update(self, *, source: jnp.ndarray, target: jnp.ndarray):
        update_forward = self._forward_update_turn or not self.ot_solver.is_bidirectional

        if update_forward:
            batch = {"source": source, "target": target}
            (
                self.ot_solver.state_f,
                self.ot_solver.state_g,
                loss,
                loss_f,
                loss_g,
                w_dist
            ) = self.ot_solver.train_step(
                self.ot_solver.state_f,
                self.ot_solver.state_g,
                batch
            )
            self._forward_update_turn = False
        else:
            batch = {"source": target, "target": source}
            (
                self.ot_solver.state_g,
                self.ot_solver.state_f,
                loss,
                loss_g,
                loss_f,
                w_dist
            ) = self.ot_solver.train_step(
                self.ot_solver.state_g,
                self.ot_solver.state_f,
                batch
            )
            self._forward_update_turn = True

        info = {
            f"{self.info_key}/forward/loss": float(loss),
            f"{self.info_key}/forward/f_potential_loss": float(loss_f),
            f"{self.info_key}/forward/g_potential_loss": float(loss_g),
            f"{self.info_key}/forward/w_dist": float(w_dist),
        }

        return info
    
    def to_dual_potentials(self):
        return self.ot_solver.to_dual_potentials()
    
    def mapping_visualization(
        self, *, source: np.ndarray, target: np.ndarray, scatter_kwargs: Optional[Dict]=None
    ) -> Tuple[plt.Figure, plt.Figure]:
        learned_potentials = self.to_dual_potentials()

        fig_forward, _ = learned_potentials.plot_ot_map(
            source=source,
            target=target,
            forward=True,
            scatter_kwargs=scatter_kwargs,
        )
        
        fig_backward, _ = learned_potentials.plot_ot_map(
            source=source,
            target=target,
            forward=False,
            scatter_kwargs=scatter_kwargs,
        )

        return (fig_forward, fig_backward)
    
    def save(self, dir_path: str):
        dir_path = Path(dir_path)
        dir_path.mkdir(exist_ok=True, parents=True)
        for attr in self._save_ot_solver_attrs:
            value = getattr(self.ot_solver, attr)
            if "state" in attr:
                _FlaxTrainStateArchiver.save(value, dir_path / attr)
            else:
                save_pickle(value, dir_path / f"{attr}.pickle")

    def load(self, dir_path: str):
        loaded_attrs = {}
        for attr in self._save_ot_solver_attrs:
            if "state" in attr:
                load_dir = dir_path / attr
                if load_dir.exists():
                    value = getattr(self.ot_solver, attr)
                    value, loaded_subattrs = _FlaxTrainStateArchiver.load(value, dir_path / attr)
                    setattr(self.ot_solver, attr, value)
                    loaded_attrs[attr] = loaded_subattrs
                else:
                    loaded_attrs[attr] = "-"
            else:
                load_path = dir_path / f"{attr}.pickle"
                if load_path.exists():
                    value = load_pickle(load_path)
                    setattr(self.ot_solver, attr, value)
                    loaded_attrs[attr] = "+"
                else:
                    loaded_attrs[attr] = "-"
        return self, loaded_attrs

class _FlaxTrainStateArchiver:
    _save_attrs: Tuple[str] = (
        "step",
        "params",
        "opt_state"
    )

    @classmethod
    def save(cls, state: FlaxTrainState, dir_path: str):
        dir_path = Path(dir_path)
        dir_path.mkdir(exist_ok=True, parents=True)
        for attr in cls._save_attrs:
            value = getattr(state, attr)
            save_pickle(value, dir_path / f"{attr}.pickle")

    @classmethod
    def load(cls, state: FlaxTrainState, dir_path: str):
        dir_path = Path(dir_path)
        attr_to_value, loaded_attrs = {}, {}
        for attr in cls._save_attrs:
            load_path = dir_path / attr
            if load_path.exists():
                value = load_pickle(load_path)
                attr_to_value[attr] = value
                loaded_attrs[attr] = "+"
            else:
                loaded_attrs[attr] = "+"
        state = state.replace(**attr_to_value)
        return state, loaded_attrs
                