import jax
import jax.numpy as jnp

from tqdm import tqdm
from loguru import logger

import wandb

from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils.save import save_buffer, load_buffer


wandb.init(project="test_jax_rl")

config = OmegaConf.load("config.yaml")

# reprodicibility
rng = jax.random.PRNGKey(config.seed)

# environment init
env = instantiate(config.environment)

# buffer init
observations, info = env.reset()
action = env.action_space.sample()
observation, reward, done, truncated, info = env.step(action)

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
    # do step in the environment
    observation_next, reward, done, truncated, info = env.step(action)

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
        observation, _ = env.reset(seed=config.seed+i)

    return observation


n_iters_collect_buffer = config.n_iters_collect_buffer
precollected_data_path = config.get("precollected_data_path")
if precollected_data_path is not None:
    state = load_buffer(state, precollected_data_path)
    n_iters_collect_buffer -= state.current_index
    n_iters_collect_buffer = max(0, n_iters_collect_buffer)
    logger.info(f"{state.current_index} samples already collected. {n_iters_collect_buffer} are left.")

observation, _, done  = *env.reset(seed=config.seed), False
for i in tqdm(range(n_iters_collect_buffer)):
    action = agent.sample_actions(observation)
    observation = do_environment_step(action)

if precollected_data_path is not None:
    state = load_buffer(state, precollected_data_path)

# training
observation, _, done  = *env.reset(seed=config.seed), False
for i in tqdm(range(config.max_steps)):
    # sample actions
    action = agent.sample_actions(observation)

    # do step in the environment
    observation = do_environment_step(action)

    # do RL optimization step
    if i >= config.start_training_after:
        rng, key = jax.random.split(rng)
        batch = buffer.sample(state, key).experience
        update_info = agent.update(batch)

        if i % config.log_every == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i)
