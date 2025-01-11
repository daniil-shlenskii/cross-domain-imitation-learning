import argparse
import warnings
from pathlib import Path

from agents.utils import instantiate_agent
from envs.collect_random_buffer import instantiate_environment
from loguru import logger
from omegaconf import OmegaConf

from utils import buffer_init, get_state_from_dict, save_json, save_pickle

from .base_agent import Agent


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
    env = instantiate_environment(eval_config.environment)

    # agent init
    agent = instantiate_agent(config.agent, env)

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
    buffer_config = config.replay_buffer
    buffer_config["fbx_buffer_config"]["min_length"] = 1
    buffer_config["fbx_buffer_config"]["max_length"] = trajs["observations"].shape[0]

    _, state = buffer_init(buffer_config, env)

    # putting roullouts into buffer
    state = get_state_from_dict(state, trajs)

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
