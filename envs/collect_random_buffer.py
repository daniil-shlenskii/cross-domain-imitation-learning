import argparse

import flashbax as fbx
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils import buffer_init, save_buffer
from utils.common_paths import DEFAULT_RANDOM_BUFFER_STORAGE_DIR
from utils.custom_types import Buffer, BufferState
from utils.fbx_buffer import get_buffer_state_size


def do_environment_step_and_update_buffer(
    *,
    env: gym.Env,
    observation: np.ndarray,
    action: np.ndarray,
    buffer: Buffer,
    state: BufferState,
    seed: int = 0,
):
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
            truncated=np.array(done or truncated),
            observations_next=np.array(observation_next),
        )
    )

    # update env if terminated
    if done or truncated:
        observation_next, _ = env.reset(seed=seed)

    return env, observation_next, state

def collect_random_buffer(
    *,
    n_iters: int,
    env: gym.Env,
    buffer: Buffer,
    state: BufferState,
    seed: int = 0,
):
    observation, _  = env.reset(seed=seed)
    for i in tqdm(range(n_iters)):
        action = env.action_space.sample()
        env, observation, state = do_environment_step_and_update_buffer(
            env=env,
            observation=observation,
            action=action,
            buffer=buffer,
            state=state,
            seed=seed+i,
        )
    return state 

def instantiate_environment(config: DictConfig):
    env = instantiate(config)
    env = RescaleAction(env, -1, 1)
    return env

def main():
    # process command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str)
    parser.add_argument("--n_iters", type=int)
    args = parser.parse_args()

    # load config
    config = OmegaConf.load(args.config)

    #
    env = instantiate_environment(config.environment)

    # buffer init
    buffer_config = {
        "_target_": "flashbax.make_item_buffer",
        "max_length": args.n_iters,
        "min_length": 1,
        "sample_batch_size": 1,
        "add_batches": False,
    }
    buffer, state = buffer_init(buffer_config, env)

    # collect random buffer
    state = collect_random_buffer(
        n_iters=args.n_iters,
        env=env,
        buffer=buffer,
        state=state,
        seed=config.get("seed", 0),
    )

    # save buffer
    save_path = DEFAULT_RANDOM_BUFFER_STORAGE_DIR / f"{config['env_name']}.pickle"
    save_buffer(state, save_path)

    print(
        f"Random Buffer with size {get_buffer_state_size(state)} is save under the following path: {save_path}"
    )

if __name__ == "__main__":
    main()
