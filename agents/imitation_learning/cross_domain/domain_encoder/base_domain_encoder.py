from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb
from agents.imitation_learning.cross_domain.domain_encoder.discriminators import \
    BaseDomainEncoderDiscriminators
from agents.imitation_learning.in_domain.gail.utils import \
    get_trajs_discriminator_logits_and_accuracy
from agents.imitation_learning.utils import (
    get_random_from_expert_buffer_state, get_state_pairs,
    get_trajectory_from_buffer, get_trajectory_from_dict,
    get_trajs_tsne_scatterplot, prepare_buffer)
from misc.gan.generator import Generator
from utils import (SaveLoadFrozenDataclassMixin, convert_figure_to_array,
                   sample_batch_jit)
from utils.custom_types import Buffer, BufferState, DataType, PRNGKey

from .utils import get_discriminators_divergence_scores, get_two_dim_data_plot


class BaseDomainEncoder(PyTreeNode, SaveLoadFrozenDataclassMixin, ABC):
    rng: PRNGKey
    target_encoder: Generator
    discriminators: BaseDomainEncoderDiscriminators
    buffer: Buffer = struct.field(pytree_node=False)
    target_random_buffer_state: BufferState = struct.field(pytree_node=False)
    source_random_buffer_state: BufferState = struct.field(pytree_node=False)
    source_expert_buffer_state: BufferState = struct.field(pytree_node=False)
    update_encoder_every: int = struct.field(pytree_node=False)
    _save_attrs: Tuple[str] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        #
        encoding_dim: int,
        target_encoder_config: DictConfig,
        discriminators_config: DictConfig,
        #
        batch_size: int,
        target_random_buffer_state_path,
        source_expert_buffer_state_path,
        #
        update_encoder_every: int = 1,
        sourse_buffer_processor_config: DictConfig = None,
        **kwargs,
    ):
        # target random buffer state init
        buffer, target_random_buffer_state = prepare_buffer(
            buffer_state_path=target_random_buffer_state_path,
            batch_size=batch_size,
        )

        # source expert buffer state init
        _, source_expert_buffer_state = prepare_buffer(
            buffer_state_path=source_expert_buffer_state_path,
            batch_size=batch_size,
            sourse_buffer_processor_config=sourse_buffer_processor_config,
        )

        # source random buffer state init
        source_random_buffer_state = get_random_from_expert_buffer_state(
            seed=seed, expert_buffer_state=source_expert_buffer_state
        )

        # discriminators init
        discriminators = instantiate(
            discriminators_config,
            seed=seed,
            encoding_dim=encoding_dim,
            _recursive_=False,
        )

        # target encoder init
        target_encoder_config = OmegaConf.to_container(target_encoder_config)
        target_encoder_config["loss_config"]["state_loss"] = discriminators.state_discriminator.state.loss_fn
        target_encoder_config["loss_config"]["policy_loss"] = discriminators.policy_discriminator.state.loss_fn

        target_dim = target_random_buffer_state.experience["observations"].shape[-1]

        target_encoder = instantiate(
            target_encoder_config,
            seed=seed,
            input_dim=target_dim,
            output_dim=encoding_dim,
            info_key="target_encoder",
            _recursive_=False,
        )

        return cls(
            rng=jax.random.key(seed),
            target_encoder=target_encoder,
            discriminators=discriminators,
            buffer=buffer,
            target_random_buffer_state=target_random_buffer_state,
            source_random_buffer_state=source_random_buffer_state,
            source_expert_buffer_state=source_expert_buffer_state,
            update_encoder_every=update_encoder_every,
            _save_attrs=(
                "target_encoder",
                "discriminators",
            ),
            **kwargs,
        )

    def __getattr__(self, item) -> Any:
        if item == "source_encoder":
            return self.target_encoder
        return super().__getattribute__(item)

    @property
    def state_discriminator(self):
        return self.discriminators.state_discriminator

    @property
    def policy_discriminator(self):
        return self.discriminators.policy_discriminator

    def pretrain_update(self):
        new_domain_encoder, info, stats_info = _pretrain_update_jit(domain_encoder=self)
        return new_domain_encoder, info, stats_info

    def update(self, target_expert_batch: DataType):
        (
            new_domain_encoder,
            target_expert_batch,
            source_random_batch,
            source_expert_batch,
            info,
            stats_info,
        ) = _update_jit(
            domain_encoder=self,
            target_expert_batch=target_expert_batch
        )
        return (
            new_domain_encoder,
            target_expert_batch,
            source_random_batch,
            source_expert_batch,
            info,
            stats_info,
        )

    @abstractmethod
    def _update_encoder(
        self,
        *,
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        pass


    def pretrain_evaluate(
        self,
        *,
        seed: int,
        two_dim_data_plot_flag: bool = False,
        convert_to_wandb_type: bool = True,
    ):
        eval_info = {}

        # get trajectories for visualization
        target_random_traj = get_trajectory_from_buffer(self.target_random_buffer_state)
        source_random_traj = get_trajectory_from_buffer(self.source_random_buffer_state)
        source_expert_traj = get_trajectory_from_buffer(self.source_expert_buffer_state)

        # preprocess trajectories
        for k in ["observations", "observations_next"]:
            target_random_traj[k] = self.target_encoder(target_random_traj[k])
            source_random_traj[k] = self.source_encoder(source_random_traj[k])
            source_expert_traj[k] = self.source_encoder(source_expert_traj[k])

        # traj_dict
        traj_dict = {
            "states": {
                "TR": target_random_traj["observations"],
                "SR": source_random_traj["observations"],
                "SE": source_expert_traj["observations"],
            },
            "state_pairs": {
                "TR": get_state_pairs(target_random_traj),
                "SR": get_state_pairs(source_random_traj),
                "SE": get_state_pairs(source_expert_traj),
            },
        }

        # get tsne scatterplots
        state_tsne_scatterplot = get_trajs_tsne_scatterplot(
            traj_dict=traj_dict["states"],
            keys_to_use=["TR", "SE"],
            seed=seed,
        )
        policy_tsne_scatterplot = get_trajs_tsne_scatterplot(
            traj_dict=traj_dict["state_pairs"],
            keys_to_use=["TR", "SR", "SE"],
            seed=seed,
        )
        if convert_to_wandb_type:
            state_tsne_scatterplot = wandb.Image(convert_figure_to_array(state_tsne_scatterplot), caption="TSNE plot of state feautures")
            policy_tsne_scatterplot = wandb.Image(convert_figure_to_array(policy_tsne_scatterplot), caption="TSNE plot of policy feautures")
        eval_info["tsne_state_scatter"] = state_tsne_scatterplot
        eval_info["tsne_policy_scatter"] = policy_tsne_scatterplot

        #
        eval_info_common_part = self._evaluate_common_part(
            seed=seed,
            traj_dict=traj_dict,
            two_dim_data_plot_flag=two_dim_data_plot_flag,
            convert_to_wandb_type=convert_to_wandb_type,
        )
        eval_info.update(eval_info_common_part)

        return eval_info

    def evaluate(
        self,
        *,
        seed: int,
        traj_dict: dict,
        two_dim_data_plot_flag: bool = False,
        convert_to_wandb_type: bool = True,
    ):
        # Get Source Random Trajectory
        source_random_traj = get_trajectory_from_buffer(self.source_random_buffer_state)
        source_random_traj = self.encode_source_batch(source_random_traj)
        traj_dict["states"]["SR"] = source_random_traj["observations"]
        traj_dict["state_pairs"]["SR"] = get_state_pairs(source_random_traj)

        eval_info = self._evaluate_common_part(
            seed=seed,
            traj_dict=traj_dict,
            two_dim_data_plot_flag=two_dim_data_plot_flag,
            convert_to_wandb_type=convert_to_wandb_type,
            TR_key="TE",
        )
        return eval_info

    def _evaluate_common_part(
        self,
        seed: int,
        traj_dict: dict,
        two_dim_data_plot_flag: bool,
        convert_to_wandb_type: bool,
        TR_key: str = "TR",
    ):
        eval_info = {}

        # get logits plot and accuracy of state_discriminator
        accuracy_dict, logits_figure_dict = get_trajs_discriminator_logits_and_accuracy(
            discriminator=self.state_discriminator,
            traj_dict=traj_dict["states"],
            keys_to_use=[TR_key, "SE"],
            discriminator_key="state",
        )
        if convert_to_wandb_type:
            for k in logits_figure_dict:
                logits_figure_dict[k] = wandb.Image(convert_figure_to_array(logits_figure_dict[k]), caption="")
        eval_info.update({**accuracy_dict, **logits_figure_dict})

        # get logits plot and accuracy of policy_discriminator
        accuracy_dict, logits_figure_dict = get_trajs_discriminator_logits_and_accuracy(
            discriminator=self.policy_discriminator,
            traj_dict=traj_dict["state_pairs"],
            keys_to_use=[TR_key, "SR", "SE"],
            discriminator_key="policy",
        )
        if convert_to_wandb_type:
            for k in logits_figure_dict:
                logits_figure_dict[k] = wandb.Image(convert_figure_to_array(logits_figure_dict[k]), caption="")
        eval_info.update({**accuracy_dict, **logits_figure_dict})

        # get divergence scores
        divergence_scores = get_discriminators_divergence_scores(domain_encoder=self, seed=seed)
        eval_info.update(divergence_scores)

        # two dimensional data plot
        if two_dim_data_plot_flag:
            two_dim_data_figure = get_two_dim_data_plot(
                traj_dict=traj_dict["states"],
                state_discriminator=self.state_discriminator,
            )
            if convert_to_wandb_type:
                two_dim_data_figure = wandb.Image(convert_figure_to_array(two_dim_data_figure), caption="Two Dim Data Plot")
            eval_info["domain_encoder/two_dim_data_plot"] = two_dim_data_figure
        return eval_info

    @jax.jit
    def encode_target_state(self, state: jnp.ndarray):
        return self.target_encoder(state)

    @jax.jit
    def encode_source_state(self, state: jnp.ndarray):
        return self.source_encoder(state)

    def sample_batches(self, rng: PRNGKey):
        new_rng, target_random_batch = sample_batch_jit(rng, self.buffer, self.target_random_buffer_state)
        new_rng, source_random_batch = sample_batch_jit(new_rng, self.buffer, self.source_random_buffer_state)
        new_rng, source_expert_batch = sample_batch_jit(new_rng, self.buffer, self.source_expert_buffer_state)
        return new_rng, target_random_batch, source_random_batch, source_expert_batch

    def encode_target_batch(self, batch):
        batch = deepcopy(batch)
        batch["observations"] = self.encode_target_state(batch["observations"])
        batch["observations_next"] = self.encode_target_state(batch["observations_next"])
        return batch

    def encode_source_batch(self, batch):
        batch = deepcopy(batch)
        batch["observations"] = self.encode_source_state(batch["observations"])
        batch["observations_next"] = self.encode_source_state(batch["observations_next"])
        return batch

    def sample_encoded_batches(self, rng: PRNGKey):
        new_rng, target_random_batch, source_random_batch, source_expert_batch = self.sample_batches(rng)
        for k in ["observations", "observations_next"]:
            target_random_batch[k] = self.encode_target_state(target_random_batch[k])
            source_random_batch[k] = self.encode_source_state(source_random_batch[k])
            source_expert_batch[k] = self.encode_source_state(source_expert_batch[k])
        return new_rng, target_random_batch, source_random_batch, source_expert_batch

@jax.jit
def _pretrain_update_jit(domain_encoder: BaseDomainEncoder):
    new_rng, target_random_batch, source_random_batch, source_expert_batch = domain_encoder.sample_batches(domain_encoder.rng)
    new_domain_encoder = domain_encoder.replace(rng=new_rng)
    (new_domain_encoder, _, _, _, info, stats_info) = _common_update_part(
        domain_encoder=new_domain_encoder,
        target_random_batch=target_random_batch,
        source_random_batch=source_random_batch,
        source_expert_batch=source_expert_batch,
    )
    return new_domain_encoder, info, stats_info

@jax.jit
def _update_jit(domain_encoder: BaseDomainEncoder, target_expert_batch: DataType):
    # sample batches
    new_rng, _, source_random_batch, source_expert_batch = domain_encoder.sample_batches(domain_encoder.rng)

    # turn target_expert_batch into target_random_batch
    target_random_batch = deepcopy(target_expert_batch)

    # TODO: 
    # new_rng, key = jax.random.split(new_rng)
    # target_random_batch["observations_next"] = jax.random.choice(
    #     key,
    #     target_expert_batch["observations_next"],
    #     shape=target_random_batch["observations_next"].shape[0],
    # )

    new_domain_encoder = domain_encoder.replace(rng=new_rng)
    return _common_update_part(
        domain_encoder=new_domain_encoder,
        target_random_batch=target_random_batch,
        source_random_batch=source_random_batch,
        source_expert_batch=source_expert_batch,
    )

def _common_update_part(
    domain_encoder: BaseDomainEncoder,
    target_random_batch: DataType,
    source_random_batch: DataType,
    source_expert_batch: DataType,
):
    # update encoder
    new_domain_encoder, info, stats_info = domain_encoder._update_encoder(
        target_random_batch=target_random_batch,
        source_random_batch=source_random_batch,
        source_expert_batch=source_expert_batch,
    )

    new_domain_encoder = jax.lax.cond(
        (domain_encoder.state_discriminator.state.step + 1) % domain_encoder.update_encoder_every == 0,
        lambda: new_domain_encoder,
        lambda: domain_encoder,
    )

    # get encoded batches
    target_random_batch = info.pop("target_random_batch")
    source_random_batch = info.pop("source_random_batch")
    source_expert_batch = info.pop("source_expert_batch")

    # update discriminators
    new_discrs, discrs_info, discrs_stats_info = domain_encoder.discriminators.update(
        target_random_batch=target_random_batch,
        source_random_batch=source_random_batch,
        source_expert_batch=source_expert_batch,
    )

    # final update
    new_domain_encoder = new_domain_encoder.replace(discriminators=new_discrs)
    info.update(discrs_info)
    stats_info.update(discrs_stats_info)

    return (
        new_domain_encoder,
        target_random_batch,
        source_random_batch,
        source_expert_batch,
        info,
        stats_info,
    )
