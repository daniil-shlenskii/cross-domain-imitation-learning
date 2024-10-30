import argparse
from pprint import pformat

import jax

import numpy as np

from tqdm import tqdm
from loguru import logger
from pathlib import Path

import wandb

from hydra.utils import instantiate
from omegaconf import OmegaConf

from evaluate import evaluate
from utils.utils import save_pickle, load_buffer


TMP_RANDOM_BUFFER_STORAGE_DIR = "_tmp_data_storage"
TMP_AGENT_STORAGE_DIR = "_tmp_agent_storage"


def init() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL agent training script"
    )
    parser.add_argument("--config_path",        type=str)
    parser.add_argument("--wandb_project_name", type=str,  default="default_wandb_project_name")
    parser.add_argument("--save_random_buffer", type=bool, default=True)
    parser.add_argument("--save_agent",         type=bool, default=True)
    return parser.parse_args()


def main(args):
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
    agent = instantiate(
        config.agent,
        observation_space=env.observation_space,
        action_space=env.action_space,
        _recursive_=False,
    )

    # load agent params if given
    load_agent_dir = Path(config_archive.get("load_agent_dir", TMP_AGENT_STORAGE_DIR)) / config.env_name
    if load_agent_dir.exists():
        loaded_keys = agent.load(load_agent_dir)
        logger.info(
            f"Agent is initialized with data under the path: {load_agent_dir}. " +
            f"Loaded keys: {loaded_keys}."
        )

    # prepare path to save agent params
    if args.save_agent:
        save_agent_dir = Path(config_archive.get("save_agent_dir", TMP_AGENT_STORAGE_DIR))
        save_agent_dir = save_agent_dir / config.env_name
        save_agent_dir.mkdir(exist_ok=True, parents=True)

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

    # collect random buffer
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

    # download random buffer if given
    n_iters_collect_buffer = config.n_iters_collect_buffer

    load_random_buffer_path = Path(config.archive.get("load_random_buffer_dir", TMP_RANDOM_BUFFER_STORAGE_DIR)) / config.env_name
    if load_random_buffer_path.exists():
        state = load_buffer(state, load_random_buffer_path)
        n_iters_collect_buffer -= state.current_index
        n_iters_collect_buffer = max(0, n_iters_collect_buffer)

        logger.info(f"Loading Random Buffer from {load_random_buffer_path}")
        logger.info(f"{state.current_index} samples already collected. {n_iters_collect_buffer} are left.")

    # collect the rest amount of the data
    logger.info("Random Buffer collecting..")

    observation, _  = env.reset(seed=config.seed)
    for i in tqdm(range(n_iters_collect_buffer)):
        action = env.action_space.sample()
        do_environment_step(action, i)

    logger.info("Random Buffer is collected.")

    # save random buffer
    if args.save_random_buffer:
        save_random_buffer_path = Path(config_archive.get("save_random_buffer_dir", TMP_RANDOM_BUFFER_STORAGE_DIR)) / config.env_name
        save_random_buffer_path.parent.mkdir(exist_ok=True, parents=True)
        save_pickle(state, save_random_buffer_path)
        logger.info(f"Random Buffer is stored under the following path: {save_random_buffer_path}")

    # training
    logger.info("Training..")

    observation, _  = env.reset(seed=config.seed)
    best_return = None
    for i in tqdm(range(config.n_iters_training)):
        # sample actions
        action = agent.sample_actions(observation[None])

        # do step in the environment
        do_environment_step(action[0], i)

        # do RL optimization step
        rng, key = jax.random.split(rng)
        batch = buffer.sample(state, key).experience
        update_info = agent.update(batch)

        # logging
        if i % config.log_every == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i)

        if i % config.eval_every == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=config.evaluation.num_episodes)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)
            if args.save_agent and (best_return is None or eval_info["return"] >= best_return):
                agent.save(save_agent_dir)
                best_return = eval_info["return"]
                logger.info(f"Best Return: {best_return}") # TODO: remove
                
    if args.save_agent:
        logger.info(
            f"Agent is stored under the path: {save_agent_dir}. " +
            f"Best Return: {np.round(best_return, 3)}"
        )

    env.close()


if __name__ == "__main__":
    args = init()
    main(args)