import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from misc.enot.utils import mapping_scatter

SOURCE_MU = jnp.array([0., 0.])
TARGET_MU1 = jnp.array([5., 5.])
TARGET_MU2 = jnp.array([-5., -5.])
SIGMA = jnp.array([.5, .5])


def loader_generator(sample_size, seed=0):
    sample_size = sample_size
    source_distr = distrax.MultivariateNormalDiag(SOURCE_MU, SIGMA)
    target_distr1 = distrax.MultivariateNormalDiag(TARGET_MU1, SIGMA)
    target_distr2 = distrax.MultivariateNormalDiag(TARGET_MU2, SIGMA)
    rng = jax.random.key(seed)

    while True:
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        source_sample = source_distr.sample(seed=k1, sample_shape=(sample_size,))
        target_sample1 = target_distr1.sample(seed=k2, sample_shape=(sample_size//2,))
        target_sample2 = target_distr2.sample(seed=k3, sample_shape=(sample_size//2,))
        target_sample = jnp.concatenate([target_sample1, target_sample2], axis=0)
        yield source_sample, target_sample

def evaluate(enot, source, target):
    target_hat = enot(source)

    fig = mapping_scatter(source, target_hat, target)
    fig = wandb.log({"scatterplot": wandb.Image(fig)})

    return {
        "cost": jax.vmap(enot.cost_fn)(source, target_hat).mean(),
        "mapping": fig
    }

def main():
    wandb.init(project="test_enot")

    config = OmegaConf.load("misc/_tests/enot/run_config.yaml")
    enot = instantiate(config.enot, source_dim=len(SIGMA), target_dim=len(SIGMA), _recursive_=False)

    loader = loader_generator(sample_size=config.batch_size)
    for i, (source_sample, target_sample) in tqdm(enumerate(loader)):
        # evaluate
        if i == 0 or (i + 1) % config.log_every == 0:
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
