import argparse
import warnings
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from ott import datasets
from tqdm import tqdm

import wandb
from utils.utils import convert_figure_to_array

TMP_ENOT_STORAGE_DIR = "archive/enot"


def init() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ..."
    )
    parser.add_argument("-w", "--ignore_warnings", action="store_true")
    return parser.parse_args()


def main():
    wandb.init(project="test_enot")

    config = OmegaConf.load("configs/_test_enot_wrapper.yaml")

    # create_dataset
    (
        train_dataloaders,
        eval_dataloaders,
        data_dim,
    ) = datasets.create_gaussian_mixture_samplers(
        name_source="square_five",
        name_target="square_four",
        valid_batch_size=config.eval_batch_size,
        train_batch_size=config.batch_size,
    )

    # enot init
    enot = instantiate(
        config.enot,
        data_dim=data_dim,
        _recursive_=False,
    )

    # load enot params if given
    enot_storage_dir = Path(TMP_ENOT_STORAGE_DIR)
    if enot_storage_dir.exists():
        _, loaded_keys = enot.load(enot_storage_dir)
        logger.info(
            f"ENOT is initialized with data under the path: {enot_storage_dir}.\n" + \
            f"Loaded keys:\n----------------\n{OmegaConf.to_yaml(loaded_keys)}"
        )

    # prepare path to save agent params
    enot_storage_dir.mkdir(exist_ok=True, parents=True)

    # training
    for i in tqdm(range(config.n_training_iters)):
        # evaluation
        if i == 0 or (i + 1) % config.eval_every == 0:
            source = jnp.asarray(next(eval_dataloaders.source_iter))
            target = jnp.asarray(next(eval_dataloaders.target_iter))
            fig_forward, fig_backward = enot.mapping_visualization(source=source, target=target)
            forward_image = wandb.Image(convert_figure_to_array(fig_forward), caption="Forward")
            backward_image = wandb.Image(convert_figure_to_array(fig_backward), caption="Backward")

            wandb.log({"evaluation/transport_map_visualization/Forward": forward_image})
            wandb.log({"evaluation/transport_map_visualization/Backward": backward_image})

        # training step
        source = jnp.asarray(next(train_dataloaders.source_iter))
        target = jnp.asarray(next(train_dataloaders.target_iter))
        update_info = enot.update(source=source, target=target)

        # logging
        if (i + 1) % config.log_every == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i)

        # saving model
        if (i + 1) % config.save_every == 0:
            enot.save(enot_storage_dir)


if __name__ == "__main__":
    args = init()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    main()