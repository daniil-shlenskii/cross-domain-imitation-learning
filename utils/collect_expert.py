import argparse
import warnings
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.evaluate import evaluate
from utils.utils import save_json, save_pickle


def init() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect rollout for the given agent."
    )
    parser.add_argument("--archive_agent_dir",     type=str)
    parser.add_argument("--num_episodes",          type=int)
    parser.add_argument("--save_rollouts_dir",     type=str, default=None)
    parser.add_argument("-w", "--ignore_warnings", action="store_true")
    return parser.parse_args()

def main(args: argparse.Namespace):
    # config init
    agent_dir = Path(args.archive_agent_dir)
    
    config = OmegaConf.load(agent_dir / "config.yaml")
    logger.info(f"\nCONFIG:\n-------\n{OmegaConf.to_yaml(config)}")

    # environment init
    eval_config = config.evaluation
    env = instantiate(eval_config.environment)
    env = RescaleAction(env, -1, 1)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=args.num_episodes)

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
    agent, loaded_keys = agent.load(agent_dir)
    logger.info(
        f"Agent is initialized with data under the path: {agent_dir}.\n" + \
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

    # collect roullouts
    logger.info("Collecting..")
    for i in tqdm(range(args.num_episodes)):
        observation, _, done, truncated = *env.reset(seed=eval_config.seed+i), False, False
        while not (done or truncated):
            # do step in the environment
            action = agent.eval_actions(observation)
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

    # save rollout and runs info
    info = {
        "num_episodes": args.num_episodes,
        "average_return": np.mean(env.return_queue),
        "average_length": np.mean(env.length_queue)
    }


    save_dir = agent_dir
    if args.save_rollouts_dir is not None:
        save_dir = Path(args.save_rollouts_dir)
    
    save_dir = save_dir / "collected_rollouts"
    save_dir.mkdir(exist_ok=True, parents=True)

    buffer_state_path = save_dir / "buffer_state.pickle"
    return_path = save_dir / "return.json"

    save_pickle(state, buffer_state_path)
    save_json(info, return_path)

    logger.info(f"Rollouts buffer_state is stored under the following path: {buffer_state_path}")
    logger.info(f"Rollouts info is stored under the following path: {return_path}")


if __name__ == "__main__":
    args = init()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    main(args)
