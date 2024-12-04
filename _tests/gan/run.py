import distrax
import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb


def main():
    wandb.init(project="test_gan")

    config = OmegaConf.load("_tests/gan/run_config.yaml")

    # reproducibility
    rng = jax.random.key(config.seed)

    # create_dataset
    dim = len(config.dataset.mu)
    mu = jnp.array([0] * dim)
    sigma = jnp.array([1] * dim)
    prior_distr = distrax.MultivariateNormalDiag(mu, sigma)

    mu = jnp.array(config.dataset.mu)
    sigma = jnp.array(config.dataset.sigma)
    target_distr = distrax.MultivariateNormalDiag(mu, sigma)

    rng, ds_key = jax.random.split(rng, 2)
    dataset = target_distr.sample(seed=ds_key, sample_shape=(config.dataset.size,))

    # gan init
    gan = instantiate(
        config.gan,
        input_dim=dataset[0].shape,
        generator_output_dim=dim,
        _recursive_=False,
    )

    # training
    for i in tqdm(range(config.n_training_iters)):
        rng, key = jax.random.split(rng)

        # evaluate model
        if i == 0 or (i + 1) % config.eval_every == 0:
            rng, key = jax.random.split(rng)
            gan_inputs = prior_distr.sample(seed=key, sample_shape=(config.eval_batch_size,)) 
            gens = gan.generator(gan_inputs)

            log_prob = target_distr.log_prob(gens).mean()
            wandb.log({"evaluation/log_prob": log_prob}, step=i)

            mu_sample = gens.mean()
            mu_loss = ((mu - mu_sample)**2).mean(-1).mean()
            wandb.log({"evaluation/mu_l2_loss": mu_loss}, step=i)

            sigma_sample = gens.mean()
            sigma_loss = ((sigma - sigma_sample)**2).mean(-1).mean()
            wandb.log({"evaluation/sigma_l2_loss": sigma_loss}, step=i)


        gan_inputs = prior_distr.sample(seed=key, sample_shape=(config.batch_size,)) 
        real_batch = target_distr.sample(seed=key, sample_shape=(config.batch_size,)) 
        update_info, stats_info = gan.update(gan_inputs, real_batch)

        # logging
        if (i + 1) % config.log_every == 0:
            for k, v in update_info.items():
                if hasattr(v, "block_until_ready"):
                    v = v.block_until_ready()
                wandb.log({f"training/{k}": v}, step=i)
            for k, v in stats_info.items():
                if hasattr(v, "block_until_ready"):
                    v = v.block_until_ready()
                wandb.log({f"training_stats/{k}": v}, step=i)


if __name__ == "__main__":
    main()
