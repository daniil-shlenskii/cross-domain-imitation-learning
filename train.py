import jax
import jax.numpy as jnp

from tqdm import tqdm
from loguru import logger
from pathlib import Path

import wandb

from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils.save import save_buffer, load_buffer


def main():
    wandb.init(project="test_jax_rl")

    config = OmegaConf.load("config.yaml")

    # reprodicibility
    rng = jax.random.PRNGKey(config.seed)

    # environment init
    env = instantiate(config.environment)

    # buffer init
    observation, info = env.reset()
    action = env.action_space.sample()
    observation, reward, _, _, _ = env.step(action)

    buffer = instantiate(config.replay_buffer)
    state = buffer.init(
        dict(
            observations=jnp.array(observation),
            actions=jnp.array(action),
            rewards=jnp.array(reward),
            observations_next=jnp.array(observation),
        )
    )

    # agent init
    agent = instantiate(
        config.agent,
        observation_space=env.observation_space,
        action_space=env.action_space,
        _recursive_=False,
    )

    # buffer collection
    def do_environment_step(action):
        nonlocal state, observation, env
    
        # do step in the environment
        observation_next, reward, done, truncated, _ = env.step(action)

        # update buffer
        state = buffer.add(
            state, 
            dict(
                observations=jnp.array(observation),
                actions=jnp.array(action),
                rewards=jnp.array(reward),
                observations_next=jnp.array(observation_next),
            )
        )

        observation = observation_next

        # update env if terminated
        if done or truncated:
            observation, _ = env.reset()

    # download buffer if needed
    n_iters_collect_buffer = config.n_iters_collect_buffer
    precollected_data_dir = Path(config.get("precollected_data_dir", "tmp_data_storage"))
    precollected_data_path = precollected_data_dir / f"{config.environment.id}.pickle"
    precollected_data_dir.mkdir(exist_ok=True)
    if precollected_data_path.exists():
        state = load_buffer(state, precollected_data_path)
        n_iters_collect_buffer -= state.current_index
        n_iters_collect_buffer = max(0, n_iters_collect_buffer)
        logger.info(f"{state.current_index} samples already collected. {n_iters_collect_buffer} are left.")

    # collect the rest of the data
    observation, _  = env.reset(seed=config.seed)
    for i in tqdm(range(n_iters_collect_buffer)):
        action = env.action_space.sample()
        do_environment_step(action)

    save_buffer(state, precollected_data_path, logger)

    # training
    observation, _  = env.reset(seed=config.seed)
    for i in tqdm(range(config.n_iters_training)):
        # sample actions
        action = agent.sample_actions(observation[None])

        # do step in the environment
        do_environment_step(action[0])

        # do RL optimization step
        rng, key = jax.random.split(rng)
        batch = buffer.sample(state, key).experience
        update_info = agent.update(batch)

        if i % config.log_every == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i)


if __name__ == "__main__":
    main()