import argparse
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict

import jax
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

import envs
import wandb
from agents.utils import instantiate_agent
from envs import (collect_random_buffer, do_environment_step_and_update_buffer,
                  instantiate_environment, register_envs)
from utils import buffer_init, get_buffer_state_size, load_buffer, save_pickle
from utils.common_paths import DEFAULT_RANDOM_BUFFER_STORAGE_DIR


def init() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL agent training script"
    )
    parser.add_argument("--config",                type=str)
    parser.add_argument("--wandb_project",         type=str, default="_default_wandb_project_name")
    parser.add_argument("--from_scratch",          action="store_true")
    parser.add_argument("-w", "--ignore_warnings", action="store_true")

    return parser.parse_args()

def get_config_archive(config: Dict, config_path: str):
    config_archive = deepcopy(config.get("archive", {}))

    default_agent_storage_dir = config_path[:-len(".yaml")]

    config_archive["agent_load_dir"] = config_archive.get("agent_load_dir", default_agent_storage_dir)
    config_archive["agent_save_dir"] = config_archive.get("agent_save_dir", default_agent_storage_dir)
    config_archive["agent_buffer_load_dir"] = config_archive.get("agent_buffer_load_dir", default_agent_storage_dir)
    config_archive["agent_buffer_save_dir"] = config_archive.get("agent_buffer_save_dir", default_agent_storage_dir)
    config_archive["random_buffer_load_dir"] = config_archive.get("random_buffer_load_dir", DEFAULT_RANDOM_BUFFER_STORAGE_DIR)
    config_archive["random_buffer_save_dir"] = config_archive.get("random_buffer_save_dir", DEFAULT_RANDOM_BUFFER_STORAGE_DIR)

    for k, v in config_archive.items():
        dir_path = Path(v)
        dir_path.mkdir(exist_ok=True, parents=True)
        config_archive[k] = dir_path

    config_archive["agent_buffer_load_path"] = config_archive["agent_buffer_load_dir"] / "buffer.pickle"
    config_archive["agent_buffer_save_path"] = config_archive["agent_buffer_save_dir"] / "buffer.pickle"
    config_archive["random_buffer_load_path"] = config_archive["random_buffer_load_dir"] / f"{config.env_name}.pickle"
    config_archive["random_buffer_save_path"] = config_archive["random_buffer_save_dir"] / f"{config.env_name}.pickle"

    return config_archive

def main(args: argparse.Namespace):
    # config init
    config = OmegaConf.load(args.config)

    ## process config part for saving/loading model
    config_archive = get_config_archive(config=config, config_path=args.config)

    ## save config into agent dir
    OmegaConf.save(config, config_archive["agent_save_dir"] / "config.yaml")

    # wandb logging init
    wandb.init(project=args.wandb_project, dir=config_archive["agent_save_dir"])

    logger.info(f"\nCONFIG:\n-------\n{OmegaConf.to_yaml(config)}")

    # reprodicibility
    rng = jax.random.PRNGKey(config.seed)

    # environment init
    env = instantiate_environment(config.environment)
    eval_env = instantiate_environment(config.evaluation.environment)

    # agent init
    agent = instantiate_agent(config.agent, env)

    ## load agent params if exist
    if not args.from_scratch and config_archive["agent_load_dir"].exists():
        agent, loaded_keys = agent.load(config_archive["agent_load_dir"])
        logger.info(
            f"Agent is initialized with data under the path: {config_archive['agent_load_dir']}.\n" + \
            f"Loaded keys:\n----------------\n{OmegaConf.to_yaml(loaded_keys)}"
        )

    # buffer init
    buffer, state = buffer_init(config.replay_buffer, env)

    # load precollected agent buffer or collect random buffer
    if not args.from_scratch and config_archive["agent_buffer_load_path"].exists():
        # load precollected agent buffer
        state = load_buffer(state, config_archive["agent_buffer_load_path"])
        logger.info(f"Loading precollected Agent Buffer from {config_archive['agent_buffer_load_path']}.")
    else:
        # collect random buffer
        ## download random buffer if given
        n_iters_collect_buffer = config.precollect_buffer_size

        if config_archive["random_buffer_load_path"].exists():
            state = load_buffer(state, config_archive["random_buffer_load_path"], size=config.precollect_buffer_size)
            n_iters_collect_buffer -= state.current_index
            n_iters_collect_buffer = max(0, n_iters_collect_buffer)

            logger.info(f"Loading Random Buffer from {config_archive['random_buffer_load_path']}.")
            logger.info(f"{state.current_index} samples already collected. {n_iters_collect_buffer} are left.")

        ## collect the rest amount of the data
        if n_iters_collect_buffer > 0:
            logger.info("Random Buffer collecting..")

            state = collect_random_buffer(
                n_iters=n_iters_collect_buffer,
                env=env,
                buffer=buffer,
                state=state,
                seed=config.seed,
            )

            logger.info("Random Buffer is collected.")

        # save random buffer
        save_pickle(state, config_archive["random_buffer_save_path"])
        logger.info(f"Random Buffer is stored under the following path: {config_archive['random_buffer_save_path']}.")

    logger.info(f"There are {get_buffer_state_size(state)} items in the Buffer.")

    # training
    logger.info("Training..")

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
                #
                **config.evaluation.get("extra_args", {})
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

        # sample actions
        if i < config.get("agent_starts_sampling_after", 0):
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(agent_sample_key, observation[None])[0]

        # do step in the environment
        env, observation, state = do_environment_step_and_update_buffer(
            env=env,
            observation=observation,
            action=action,
            buffer=buffer,
            state=state,
            seed=config.seed+i,
        )

        # do RL optimization step
        batch = buffer.sample(state, buffer_sample_key).experience
        agent, update_info, stats_info = agent.update(batch)

        # logging
        if (i + 1) % config.log_every == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i)
            for k, v in stats_info.items():
                wandb.log({f"training_stats/{k}": v}, step=i)

        # save model
        if (i + 1) % config.save_every == 0:
            agent.save(config_archive["agent_save_dir"])
            save_pickle(state, config_archive["agent_buffer_save_path"])

    logger.info(f"Agent is stored under the path: {config_archive['agent_save_dir']}")

    env.close()


if __name__ == "__main__":
    args = init()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    register_envs()

    main(args)
