import gymnasium as gym
import flashbax as fbx

import jax
import jax.numpy as jnp

from hydra.utils import instantiate
from omegaconf import OmegaConf


config = OmegaConf.load("config.yaml")
print(type(config))

env = instantiate(config.environment)

# buffer init
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

buffer = instantiate(config.replay_buffer)
state = buffer.init(
    dict(
        obs=jnp.array(obs),
        reward=jnp.array(reward),
        obs_next=jnp.array(obs),
    )
)
###

obs, _, done  = *env.reset(seed=config.seed), False
for i in range(config.max_steps):


    # sample actions
    if i < config.start_training_after:
        action = env.action_space.sample()
    else:
        action = agent.sample_actions(obs)

    # do step in the environment
    obs_next, reward, terminated, truncated, info = env.step(action)

    # update buffer
    state = buffer.add(
        state, 
        dict(
            obs=jnp.array(obs),
            reward=jnp.array(reward),
            obs_next=jnp.array(obs_next),
        )
    )
    obs = obs_next

    if terminated or truncated:
        obs, info = env.reset(...)

    # do RL optimization step
    if i >= config.start_training_after:
        batch = buffer.sample(state, ...)

env.close()

