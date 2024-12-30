import argparse
import warnings
from pathlib import Path

import numpy as np
from gymnasium.wrappers import RescaleAction
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from agents.base_agent import Agent
from utils import save_json, save_pickle


def init() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect rollout for the given agent."
    )
    parser.add_argument("--agent_dir",             type=str)
    parser.add_argument("--num_episodes",          type=int)
    parser.add_argument("--save_rollouts_dir",     type=str, default=None)
    parser.add_argument("-w", "--ignore_warnings", action="store_true")
    return parser.parse_args()

def main(args: argparse.Namespace):
    # agent config init
    agent_dir = Path(args.agent_dir)
    
    config = OmegaConf.load(agent_dir / "config.yaml")
    logger.info(f"\nCONFIG:\n-------\n{OmegaConf.to_yaml(config)}")

    # environment init
    eval_config = config.evaluation
    env = instantiate(eval_config.environment)
    env = RescaleAction(env, -1, 1)

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

    # collect rollouts
    logger.info("Collecting..")
    info, trajs = Agent.evaluate(
        agent,
        seed=eval_config.seed,
        env=env,
        num_episodes=args.num_episodes,
        return_trajectories=True,
    )

    # buffer init
    observation, _ = env.reset()
    action = env.action_space.sample()
    observation, reward, done, truncated, _ = env.step(action)

    buffer_config = config.replay_buffer
    buffer_config["fbx_buffer_config"]["min_length"] = 0
    buffer_config["fbx_buffer_config"]["max_length"] = trajs["observations"].shape[0]

    buffer = instantiate(config.replay_buffer, _recursive_=False)
    state = buffer.init(
        dict(
            observations=np.array(observation),
            actions=np.array(action),
            rewards=np.array(reward),
            dones=np.array(done),
            truncated=np.array(truncated or done),
            observations_next=np.array(observation),
        )
    )

    # putting roullouts into buffer
    new_state_exp = trajs
    for k, v in new_state_exp.items():
        new_state_exp[k] = v[None]
    state = state.replace(
        experience=new_state_exp,
        is_full=True,
        current_index=0,
    )

    # save rollout and runs info
    info.update({"num_episodes": args.num_episodes})

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
