import argparse
import os
import warnings
from typing import Dict

import jax
import numpy as np
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
    config.archive.agent_buffer_load_dir = config.archive.get("agent_buffer_load_dir", default_agent_storage_dir)
    config.archive.agent_buffer_save_dir = config.archive.get("agent_buffer_save_dir", default_agent_storage_dir)
    config.archive.random_buffer_load_dir = config.archive.get("random_buffer_load_dir", DEFAULT_RANDOM_BUFFER_STORAGE_DIR)
    config.archive.random_buffer_save_dir = config.archive.get("random_buffer_save_dir", DEFAULT_RANDOM_BUFFER_STORAGE_DIR)

    for path in config.archive.values():
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)

    config.archive.agent_buffer_load_path = os.path.join(config.archive.agent_buffer_load_dir, "buffer.pickle")
    config.archive.agent_buffer_save_path = os.path.join(config.archive.agent_buffer_save_dir, "buffer.pickle")
    config.archive.random_buffer_load_path = os.path.join(config.archive.random_buffer_load_dir, f"{config.env_name}.pickle")
    config.archive.random_buffer_save_path = os.path.join(config.archive.random_buffer_save_dir, f"{config.env_name}.pickle")

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

    # wandb logging init
    wandb_run = wandb.init(project=wandb_project, dir=config.archive.agent_save_dir)

    # reprodicibility
    rng = jax.random.PRNGKey(config.seed)

    # environment init
    env = instantiate_environment(config.environment)
    eval_env = instantiate_environment(config.evaluation.environment)

    # buffer init
    buffer, state = buffer_init(config.replay_buffer, env)

    # load precollected agent buffer or collect random buffer
    if not from_scratch and os.path.exists(config.archive.agent_buffer_load_path):
        state = load_buffer(state, config.archive.agent_buffer_load_path)
        logger.info(f"Loading precollected Agent Buffer from {config.archive.agent_buffer_load_path}.")
    else:
        # download random buffer if given
        if os.path.exists(config.archive.random_buffer_load_path):
            state = load_buffer(state, config.archive.random_buffer_load_path, size=config.precollect_buffer_size)
            logger.info(f"Loading Random Buffer from {config.archive.random_buffer_load_path}.")
            logger.info(f"{get_buffer_state_size(state)} samples are loaded.")

        # collect the rest amount of the data
        n_iters_collect_buffer = max(0, config.precollect_buffer_size - get_buffer_state_size(state))
        if n_iters_collect_buffer > 0:
            logger.info("Random Buffer collecting..")
            state = collect_random_buffer(
                n_iters=n_iters_collect_buffer,
                env=env,
                buffer=buffer,
                state=state,
                seed=config.seed,
            )
            # save random buffer
            save_pickle(state, config.archive.random_buffer_save_path)
            logger.info(f"Random Buffer is stored under the following path: {config.archive.random_buffer_save_path}.")
    logger.info(f"There are {get_buffer_state_size(state)} items in the Buffer.")

    # agent init
    agent = instantiate_agent(config.agent, env)

    ## load agent params if exist
    if not from_scratch and os.path.exists(config.archive.agent_load_dir):
        agent, loaded_keys = agent.load(config.archive.agent_load_dir)
        logger.info(
            f"Agent is initialized with data under the path: {config.archive.agent_load_dir}.\n" + \
            f"Loaded keys:\n----------------\n{OmegaConf.to_yaml(loaded_keys)}"
        )

    # pre-training
    logger.info("Pre-Training..")
    for i in tqdm(range(config.get("n_iters_pretraining", 0))):
        # reproducibility
        rng, buffer_sample_key = jax.random.split(rng, 2)

        # evaluate model
        if i == 0 or (i + 1) % config.eval_every == 0:
            eval_info = agent.evaluate(
                seed=config.evaluation.seed,
                env=eval_env,
                num_episodes=config.evaluation.num_episodes,
                **config.evaluation.get("extra_args", {})
            )
            for k, v in eval_info.items():
                wandb_run.log({f"evaluation/{k}": v}, step=i)

        # update agent
        batch = buffer.sample(state, buffer_sample_key).experience
        agent, update_info, stats_info = agent.pretrain_update(batch)

        # logging
        if (i + 1) % config.log_every == 0:
            for k, v in update_info.items():
                wandb_run.log({f"training/{k}": v}, step=i)
            for k, v in stats_info.items():
                wandb_run.log({f"training_stats/{k}": v}, step=i)

        # save model
        if (i + 1) % config.save_every == 0:
            agent.save(config.archive.agent_save_dir)
            save_pickle(state, config.archive.agent_buffer_save_path)

    # training
    logger.info("Training..")
    returns_history = []

    observation, _  = env.reset(seed=config.seed)
    for i in tqdm(range(config.n_iters_training)):
        # reproducibility
        rng, agent_sample_key, buffer_sample_key = jax.random.split(rng, 3)

        # evaluate model
        if i == 0 or (i + 1) % config.eval_every == 0:
            eval_info = agent.evaluate(
                seed=config.evaluation.seed,
                env=eval_env,
                num_episodes=config.evaluation.num_episodes,
                **config.evaluation.get("extra_args", {})
            )
            for k, v in eval_info.items():
                wandb_run.log({f"evaluation/{k}": v}, step=i)
            returns_history.append(eval_info["return"])

        # sample actions
        action = agent.sample_actions(agent_sample_key, observation)

        # do step in the environment
        env, observation, state = do_environment_step_and_update_buffer(
            env=env,
            observation=observation,
            action=action,
            buffer=buffer,
            state=state,
            seed=config.seed+i,
        )

        # update agent
        batch = buffer.sample(state, buffer_sample_key).experience
        agent, update_info, stats_info = agent.update(batch)

        # logging
        if (i + 1) % config.log_every == 0:
            for k, v in update_info.items():
                wandb_run.log({f"training/{k}": v}, step=i)
            for k, v in stats_info.items():
                wandb_run.log({f"training_stats/{k}": v}, step=i)

        # save model
        if (i + 1) % config.save_every == 0:
            agent.save(config.archive.agent_save_dir)
            save_pickle(state, config.archive.agent_buffer_save_path)

    logger.info(f"Agent is stored under the path: {config.archive.agent_save_dir}")
    env.close()
    wandb_run.finish(quiet=True)

    return returns_history

def optuna_function(config: DictConfig):
    returns_history = main(config)
    return np.max(returns_history)


if __name__ == "__main__":
    # parse command line
    args = init()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    # load config
    config = OmegaConf.load(args.config)
    logger.info(f"\nCONFIG:\n-------\n{OmegaConf.to_yaml(config)}")

    main(
        config,
        default_agent_storage_dir=args.config[:-len(".yaml")],
        from_scratch=args.from_scratch,
        wandb_project=args.wandb_project,
    )
