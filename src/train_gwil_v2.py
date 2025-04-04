import argparse
import os
import warnings
from typing import Dict

import jax
import numpy as np
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from agents.utils import instantiate_agent
from envs import (collect_random_buffer, do_environment_step_and_update_buffer,
                  instantiate_environment)
from utils import buffer_init, get_buffer_state_size, load_buffer, save_pickle
from utils.common_paths import (DEFAULT_RANDOM_BUFFER_STORAGE_DIR,
                                DEFAULT_WANDB_PROJECT)


def init() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL agent training script"
    )
    parser.add_argument("--config",                type=str)
    parser.add_argument("--wandb_project",         type=str, default="_default_wandb_project_name")
    parser.add_argument("--from_scratch",          action="store_true")
    parser.add_argument("-w", "--ignore_warnings", action="store_true")

    return parser.parse_args()

def process_archivation_paths(config: Dict, default_agent_storage_dir: str):
    config.archive = config.get("archive", {})

    config.archive.agent_load_dir = config.archive.get("agent_load_dir", default_agent_storage_dir)
    config.archive.agent_save_dir = config.archive.get("agent_save_dir", default_agent_storage_dir)

    for path in config.archive.values():
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)

    return config

def main(
    config: DictConfig,
    *,
    default_agent_storage_dir="._default_agent_storage_dir",
    from_scratch: bool = True,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
):
    # set missing paths for logging and save config
    config = process_archivation_paths(config=config, default_agent_storage_dir=default_agent_storage_dir)
    OmegaConf.save(config, os.path.join(config.archive.agent_save_dir, "config.yaml"))
    logger.info(f"\nCONFIG:\n-------\n{OmegaConf.to_yaml(config)}")

    # wandb logging init
    wandb_run = wandb.init(project=wandb_project, dir=config.archive.agent_save_dir)

    # agent init
    agent = instantiate(config.agent, _recursive_=False)

    ## load agent params if exist
    if not from_scratch and os.path.exists(config.archive.agent_load_dir):
        agent, loaded_keys = agent.load(config.archive.agent_load_dir)
        logger.info(
            f"Agent is initialized with data under the path: {config.archive.agent_load_dir}.\n" + \
            f"Loaded keys:\n----------------\n{OmegaConf.to_yaml(loaded_keys)}"
        )

    # training
    logger.info("Training..")
    agent.train(
        random_buffer_size=config.train.random_buffer_size,
        n_pretrain_iters=config.train.n_pretrain_iters,
        n_train_iters=config.train.n_train_iters,
        update_target_learner_every=config.train.update_target_learner_every,
        update_source_learner_every=config.train.update_source_learner_every,
        log_every=config.logging.log_every,
        save_every=config.logging.save_every,
        eval_every=config.logging.eval_every,
        n_eval_episodes=config.logging.n_eval_episodes,
        wandb_run=wandb_run,
    )

if __name__ == "__main__":
    # parse command line
    args = init()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    # load config
    main_config = OmegaConf.load(args.config)
    fixed_learners_config = OmegaConf.load("configs/gwil_v2/fixed_learners_configs.yaml")
    config = OmegaConf.merge(main_config, fixed_learners_config)

    main(
        config,
        default_agent_storage_dir=args.config[:-len(".yaml")],
        from_scratch=args.from_scratch,
        wandb_project=args.wandb_project,
    )
