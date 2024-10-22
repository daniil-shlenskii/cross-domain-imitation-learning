import gymnasium as gym
import flashbax as fbx

import jax
import jax.numpy as jnp

from hydra.utils import instantiate
from omegaconf import OmegaConf


config = OmegaConf.load("config.yaml")
print(type(config))

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

# training
observation, _, done  = *env.reset(seed=config.seed), False
for i in range(config.max_steps):
    print(i)
    # sample actions
    if i < config.start_training_after:
        actions = env.action_space.sample()
    else:
        actions = agent.sample_actions(observations)

    # do step in the environment
    observations_next, rewards, done, truncated, info = env.step(actions)

    # update buffer
    state = buffer.add(
        state, 
        dict(
            observations=jnp.array(observation),
            actions=jnp.array(action),
            rewards=jnp.array(reward),
            observations_next=jnp.array(observation),
        )
    )
    observations = observations_next

    # update env if terminated
    if done or truncated:
        observations, info = env.reset(seed=config.seed+i)

    # do RL optimization step
    if i >= config.start_training_after:
        rng, key = jax.random.split(rng)
        batch = buffer.sample(state, key)

env.close()

print("finish")