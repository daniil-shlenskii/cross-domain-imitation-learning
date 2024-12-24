import argparse
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict

from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb


def init() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training script"
    )
    parser.add_argument("--config",                type=str)
    parser.add_argument("--wandb_project",         type=str, default="_default_wandb_project_name")
    parser.add_argument("--from_scratch",          action="store_true")
    parser.add_argument("-w", "--ignore_warnings", action="store_true")

    return parser.parse_args()

def get_config_archive(config: Dict, config_path: str):
    config_archive = deepcopy(config.get("archive", {}))

    default_storage_dir = config_path[:-len(".yaml")]

    config_archive["load_dir"] = Path(config_archive.get("load_dir", default_storage_dir)) / "domain_encoder"
    config_archive["save_dir"] = Path(config_archive.get("save_dir", default_storage_dir)) / "domain_encoder"

    config_archive["load_dir"].mkdir(exist_ok=True, parents=True)
    config_archive["save_dir"].mkdir(exist_ok=True, parents=True)

    return config_archive

def main(args: argparse.Namespace):
    # config init
    config = OmegaConf.load(args.config)

    ## process config part for saving/loading model
    config_archive = get_config_archive(config=config, config_path=args.config)

    ## save config into model dir
    OmegaConf.save(config, config_archive["save_dir"] / "config.yaml")

    # wandb logging init
    wandb.init(project=args.wandb_project, dir=config_archive["save_dir"])

    logger.info(f"\nCONFIG:\n-------\n{OmegaConf.to_yaml(config)}")

    # model init
    model = instantiate(config.model, _recursive_=False)

    ## load model gent params if exist
    if not args.from_scratch and config_archive["load_dir"].exists():
        model, loaded_keys = model.load(config_archive["load_dir"])
        logger.info(
            f"Model is initialized with data under the path: {config_archive['load_dir']}.\n" + \
            f"Loaded keys:\n----------------\n{OmegaConf.to_yaml(loaded_keys)}"
        )

    # training
    logger.info("Training..")

    for i in tqdm(range(config.n_iters_training)):
        # evaluate model
        if i == 0 or (i + 1) % config.eval_every == 0:
            eval_info = model.evaluate(config.seed)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

        # optimization step
        model, update_info, stats_info = model.pretrain_update()

        # logging
        if (i + 1) % config.log_every == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i)
            for k, v in stats_info.items():
                wandb.log({f"training_stats/{k}": v}, step=i)

        # save model
        if (i + 1) % config.save_every == 0:
            model.save(config_archive["save_dir"])

    logger.info(f"Model is stored under the path: {config_archive['save_dir']}")


if __name__ == "__main__":
    args = init()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    main(args)
