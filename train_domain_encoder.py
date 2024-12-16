import argparse
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict

import jax
import numpy as np
from gymnasium.wrappers import RescaleAction
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from utils import get_buffer_state_size, load_buffer, save_pickle


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
    default_random_buffer_storate_dir = "._tmp_archive_dir/random_buffers"

    config_archive["agent_load_dir"] = config_archive.get("agent_load_dir", default_agent_storage_dir) + "/domain_encoder"
    config_archive["agent_save_dir"] = config_archive.get("agent_save_dir", default_agent_storage_dir) + "/domain_encoder"
    config_archive["agent_buffer_load_dir"] = config_archive.get("agent_buffer_load_dir", default_agent_storage_dir)
    config_archive["agent_buffer_save_dir"] = config_archive.get("agent_buffer_save_dir", default_agent_storage_dir)
    config_archive["random_buffer_load_dir"] = config_archive.get("random_buffer_load_dir", default_random_buffer_storate_dir)
    config_archive["random_buffer_save_dir"] = config_archive.get("random_buffer_save_dir", default_random_buffer_storate_dir)

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
    env = instantiate(config.environment)
    env = RescaleAction(env, -1, 1)

    # agent init
    agent = instantiate(config.agent, _recursive_=False)

    ## load agent params if exist
    if not args.from_scratch and config_archive["agent_load_dir"].exists():
        agent, loaded_keys = agent.load(config_archive["agent_load_dir"])
        logger.info(
            f"Agent is initialized with data under the path: {config_archive['agent_load_dir']}.\n" + \
            f"Loaded keys:\n----------------\n{OmegaConf.to_yaml(loaded_keys)}"
        )

    # buffer init
    observation, _ = env.reset()
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)

    buffer = instantiate(config.replay_buffer, _recursive_=False)
    state = buffer.init(
        dict(
            observations=np.array(observation),
            actions=np.array(action),
            rewards=np.array(reward),
            dones=np.array(done),
            observations_next=np.array(observation),
        )
    )

    ## load precollected agent buffer or collect random buffer
    def do_environment_step(action, i):
        nonlocal env, state, observation

        # do step in the environment
        observation_next, reward, done, truncated, _ = env.step(action)

        # update buffer
        state = buffer.add(
            state,
            dict(
                observations=np.array(observation),
                actions=np.array(action),
                rewards=np.array(reward),
                dones=np.array(done),
                observations_next=np.array(observation_next),
            )
        )

        observation = observation_next

        # update env if terminated
        if done or truncated:
            observation, _ = env.reset(seed=config.seed + i)

    ## precollect agent buffer
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

            observation, _  = env.reset(seed=config.seed)
            for i in tqdm(range(n_iters_collect_buffer)):
                action = env.action_space.sample()
                do_environment_step(action, i)

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

        # sample actions
        action = env.action_space.sample()

        # do step in the environment
        do_environment_step(action, i)

        # do RL optimization step
        batch = buffer.sample(state, buffer_sample_key).experience
        agent, _, _, _, _, update_info, stats_info = agent.update(batch)

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

    main(args)
