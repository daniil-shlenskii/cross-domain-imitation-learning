import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from misc.enot.utils import mapping_scatter
from utils import convert_figure_to_array, load_pickle


def loader_generator(sample_size, seed=0, ds_name="gaussian"):
    rng = jax.random.key(seed)

    if ds_name == "gaussian":
        SOURCE_MU = jnp.array([0., 0.])
        TARGET_MU1 = jnp.array([5., 5.])
        TARGET_MU2 = jnp.array([-5., -5.])
        SIGMA = jnp.array([.5, .5])
        source_distr = distrax.MultivariateNormalDiag(SOURCE_MU, SIGMA)
        target_distr1 = distrax.MultivariateNormalDiag(TARGET_MU1, SIGMA)
        target_distr2 = distrax.MultivariateNormalDiag(TARGET_MU2, SIGMA)

        while True:
            rng, k1, k2, k3 = jax.random.split(rng, 4)
            source_sample = source_distr.sample(seed=k1, sample_shape=(sample_size,))
            target_sample1 = target_distr1.sample(seed=k2, sample_shape=(sample_size//2,))
            target_sample2 = target_distr2.sample(seed=k3, sample_shape=(sample_size//2,))
            target_sample = jnp.concatenate([target_sample1, target_sample2], axis=0)
            yield source_sample, target_sample
    elif ds_name == "lines":
        ds_size = 20_000
        ds = jnp.linspace(0, 1, ds_size)
        while True:
            rng, k1, k2 = jax.random.split(rng, 3)
            idcs1 = jax.random.choice(k1, ds_size, shape=(sample_size,))
            idcs2 = jax.random.choice(k2, ds_size, shape=(sample_size,))
            source_sample = jnp.stack([ds[idcs1], jnp.zeros(sample_size)], axis=1)
            target_sample = jnp.stack([ds[idcs2], jnp.ones(sample_size)], axis=1)
            yield source_sample, target_sample
    elif ds_name == "temporal":
        ds_size = 20_000
        ds = jnp.linspace(0, 1, ds_size)
        while True:
            rng, k1, k2 = jax.random.split(rng, 3)
            idcs1 = jax.random.choice(k1, ds_size//2, shape=(sample_size,))
            idcs2 = jax.random.choice(k2, ds_size, shape=(sample_size,))
            source_sample = jnp.stack([jnp.zeros(sample_size), ds[idcs1], idcs1/ds_size], axis=1)
            target_sample = jnp.stack([-ds[idcs2], jnp.zeros(sample_size), idcs2/ds_size], axis=1)
            yield source_sample, target_sample
    elif ds_name == "lines_double":
        ds_size = 20_000
        ds = jnp.linspace(0, 1, ds_size)
        while True:
            rng, k1, k2 = jax.random.split(rng, 3)
            idcs1 = jax.random.choice(k1, ds_size, shape=(sample_size,))
            idcs2 = jax.random.choice(k2, ds_size, shape=(sample_size,))
            source_sample1 = jnp.stack([ds[idcs1[:sample_size//2]], -jnp.ones(sample_size//2)], axis=1)
            source_sample2 = jnp.stack([ds[idcs1[sample_size//2:]] + 1., -2*jnp.ones(sample_size//2)], axis=1)
            target_sample1 = jnp.stack([ds[idcs1[:sample_size//2]], jnp.ones(sample_size//2)], axis=1)
            target_sample2 = jnp.stack([ds[idcs1[sample_size//2:]] + 1, 2*jnp.ones(sample_size//2)], axis=1)
            source_sample = jnp.concatenate([source_sample1, source_sample2], axis=0)
            target_sample = jnp.concatenate([target_sample1, target_sample2], axis=0)
            yield source_sample, target_sample
    elif ds_name == "agent":
        expert_ds_size = 20_000
        agent_ds_frac = 0.1
        agent_ds_size = int(expert_ds_size * agent_ds_frac)
        expert_ds = jnp.linspace(0, 1, expert_ds_size)
        agent_ds = jnp.linspace(0, agent_ds_frac, agent_ds_size)
        while True:
            rng, k1, k2 = jax.random.split(rng, 3)
            idcs1 = jax.random.choice(k1, agent_ds_size, shape=(sample_size,))
            idcs2 = jax.random.choice(k2, expert_ds_size, shape=(sample_size,))
            source_sample = jnp.stack([agent_ds[idcs1], jnp.zeros(sample_size)], axis=1)
            target_sample = jnp.stack([expert_ds[idcs2], jnp.ones(sample_size)], axis=1)
            yield source_sample, target_sample
    elif "from_buffer" in ds_name:
        source_path = "buffers/Hopper/Hopper_random_buffer.pickle"
        target_path = "buffers/Hopper/Hopper_expert_buffer.pickle"

        source_buffer = load_pickle(source_path)
        target_buffer = load_pickle(target_path)
        if source_buffer.is_full:
            source_buffer = {k: v[0] for k, v in source_buffer.experience.items()}
        else:
            source_buffer = {k: v[0, :source_buffer.current_index] for k, v in source_buffer.experience.items()}
        if target_buffer.is_full:
            target_buffer = {k: v[0] for k, v in target_buffer.experience.items()}
        else:
            target_buffer = {k: v[0, :target_buffer.current_index] for k, v in target_buffer.experience.items()}

        if ds_name == "from_buffer_states":
            source_ds = source_buffer["observations"]
            target_ds = target_buffer["observations"]
        elif ds_name == "from_buffer_state_pairs":
            source_ds = jnp.concatenate([source_buffer["observations"], source_buffer["observations_next"]], axis=1)
            target_ds = jnp.concatenate([target_buffer["observations"], target_buffer["observations_next"]], axis=1)

        while True:
            rng, k1, k2 = jax.random.split(rng, 3)
            source_sample = jax.random.choice(k1, source_ds, shape=(sample_size,))
            target_sample = jax.random.choice(k2, target_ds, shape=(sample_size,))
            yield source_sample, target_sample

def evaluate(enot, source, target):
    target_hat = enot(source)
    fig = mapping_scatter(source, target_hat, target)
    fig = wandb.Image(convert_figure_to_array(fig))

    P = enot.cost_fn.proj_matrix
    P_source = jax.vmap(lambda x: P @ x)(source)
    figP = mapping_scatter(source, P_source, target)
    figP = wandb.Image(convert_figure_to_array(figP))

    return {
        "cost": enot.cost(source, target_hat).mean(),
        "mapping": fig,
        "Pmapping": figP,
    }

def main():
    wandb.init(project="test_enot")

    config = OmegaConf.load("misc/_tests/enot/run_config.yaml")
    loader = loader_generator(sample_size=config.batch_size, ds_name=config.ds_name)

    source_sample, target_sample = next(loader)
    enot = instantiate(config.enot, source_dim=source_sample.shape[-1], target_dim=target_sample.shape[-1], _recursive_=False)

    for i, (source_sample, target_sample) in tqdm(enumerate(loader)):
        # evaluate
        if i == 0 or (i + 1) % config.eval_every == 0:
            eval_info = evaluate(enot, source_sample, target_sample)
            for k, v in eval_info.items():
                wandb.log({f"eval/{k}": v}, step=i)

        enot, update_info, stats_info = enot.update(target=target_sample, source=source_sample)

        # logging
        if (i + 1) % config.log_every == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i)
            for k, v in stats_info.items():
                wandb.log({f"training_stats/{k}": v}, step=i)

        if (i + 1) == config.n_training_iters:
            break


if __name__ == "__main__":
    main()
