import argparse
import warnings
from pathlib import Path
from pprint import pformat

import jax
import numpy as np
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from utils.evaluate import evaluate
from utils.utils import load_buffer, save_pickle

TMP_RANDOM_BUFFER_STORAGE_DIR = "_tmp_data_storage"
TMP_AGENT_STORAGE_DIR = "_tmp_agent_storage"
AGENT_BUFFER_FILENAME = "buffer"


def init() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL agent training script"
    )
    parser.add_argument("--config_path",           type=str)
    parser.add_argument("--wandb_project_name",    type=str, default="_default_wandb_project_name")
    parser.add_argument("--from_scratch",          action="store_true")
    parser.add_argument("-w", "--ignore_warnings", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace):
    wandb.init(project=args.wandb_project_name)

    config = OmegaConf.load(args.config_path)
    config_archive = config.get("archive", {})

    logger.info(f"\nCONFIG:\n-------\n{OmegaConf.to_yaml(config)}")

    # reprodicibility
    rng = jax.random.PRNGKey(config.seed)

    # environment init
    env = instantiate(config.environment)
    eval_env = instantiate(config.evaluation.environment)

    # agent init
    observation_space = env.observation_space
    action_space = env.action_space

    observation_dim = observation_space.sample().shape[-1]
    action_dim = action_space.sample().shape[-1]
    low, high = action_space.low, action_space.high
    if np.any(low == -1) or np.any(high == 1):
        low, high = None, None
        
    agent = instantiate(
        config.agent,
        observation_dim=observation_dim,
        action_dim=action_dim,
        low=low,
        high=high,
        _recursive_=False,
    )

    # load agent params if given
    agent_load_dir = Path(config_archive.get("agent_load_dir", TMP_AGENT_STORAGE_DIR)) / config.env_name
    if not args.from_scratch and agent_load_dir.exists():
        _, loaded_keys = agent.load(agent_load_dir)
        logger.info(
            f"Agent is initialized with data under the path: {agent_load_dir}.\n" + \
            f"Loaded keys:\n----------------\n{OmegaConf.to_yaml(loaded_keys)}"
        )

    # prepare path to save agent params
    agent_save_dir = Path(config_archive.get("agent_save_dir", TMP_AGENT_STORAGE_DIR)) / config.env_name
    agent_save_dir.mkdir(exist_ok=True, parents=True)

    # save agent config
    OmegaConf.save(config, agent_save_dir / "config.yaml")

    # prepare path to save agent buffer
    agent_buffer_load_dir = Path(config_archive.get("agent_buffer_load_dir", TMP_AGENT_STORAGE_DIR)) / config.env_name
    agent_buffer_load_dir.mkdir(exist_ok=True, parents=True)
    agent_buffer_load_path = AGENT_BUFFER_FILENAME

    # buffer init
    observation, _ = env.reset()
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)

    buffer = instantiate(config.replay_buffer)
    state = buffer.init(
        dict(
            observations=np.array(observation),
            actions=np.array(action),
            rewards=np.array(reward),
            dones=np.array(done),
            observations_next=np.array(observation),
        )
    )

    # load precollected agent buffer or collect random buffer
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

    agent_buffer_load_path = Path(config_archive.get("agent_buffer_load_dir", TMP_AGENT_STORAGE_DIR)) / config.env_name / "buffer"
    if not args.from_scratch and agent_buffer_load_path.exists():
        # load precollected agent buffer
        state = load_buffer(state, agent_buffer_load_path)
        logger.info(f"Loading precollected Agent Buffer from {agent_buffer_load_path}.")
    else:
        # collect random buffer
        # download random buffer if given
        n_iters_collect_buffer = config.precollect_buffer_size

        random_buffer_load_path = Path(config_archive.get("random_buffer_load_dir", TMP_RANDOM_BUFFER_STORAGE_DIR)) / config.env_name
        if random_buffer_load_path.exists():
            state = load_buffer(state, random_buffer_load_path)
            n_iters_collect_buffer -= state.current_index
            n_iters_collect_buffer = max(0, n_iters_collect_buffer)

            logger.info(f"Loading Random Buffer from {random_buffer_load_path}.")
            logger.info(f"{state.current_index} samples already collected. {n_iters_collect_buffer} are left.")

        # collect the rest amount of the data
        if n_iters_collect_buffer > 0:
            logger.info("Random Buffer collecting..")

            observation, _  = env.reset(seed=config.seed)
            for i in tqdm(range(n_iters_collect_buffer)):
                action = env.action_space.sample()
                do_environment_step(action, i)

            logger.info("Random Buffer is collected.")

        # save random buffer
        random_buffer_save_path = Path(config_archive.get("random_buffer_save_dir", TMP_RANDOM_BUFFER_STORAGE_DIR)) / config.env_name
        random_buffer_save_path.parent.mkdir(exist_ok=True, parents=True)
        save_pickle(state, random_buffer_save_path)
        logger.info(f"Random Buffer is stored under the following path: {random_buffer_save_path}.")

    logger.info(f"There are {state.current_index} items in the Buffer.")

    # training
    logger.info("Training..")

    observation, _  = env.reset(seed=config.seed)
    for i in tqdm(range(config.n_iters_training)):
        # evaluate model
        if i == 0 or (i + 1) % config.eval_every == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=config.evaluation.num_episodes,
                seed=config.evaluation.seed,
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

        # sample actions
        action = agent.sample_actions(observation[None])

        # do step in the environment
        do_environment_step(action[0], i)

        # do RL optimization step
        rng, key = jax.random.split(rng)
        batch = buffer.sample(state, key).experience
        update_info, stats_info = agent.update(batch)

        # logging
        if (i + 1) % config.log_every == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i)
            for k, v in stats_info.items():
                wandb.log({f"training_stats/{k}": v}, step=i)

        # save model
        if (i + 1) % config.save_every == 0:
            agent.save(agent_save_dir)
            save_pickle(state, agent_buffer_load_path)

    logger.info(f"Agent is stored under the path: {agent_save_dir}.")

    env.close()


if __name__ == "__main__":
    args = init()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    main(args)
