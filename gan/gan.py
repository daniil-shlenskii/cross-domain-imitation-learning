import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from gan.discriminator import Discriminator
from gan.generator import Generator
from utils.types import Params, PRNGKey
from utils.utils import instantiate_optimizer


class GAN:
    @classmethod
    def create(
        cls,
        *,
        seed: int,
        input_sample: jnp.ndarray,
        generator_output_dim: int,
        #
        generator_config: DictConfig,
        discriminator_config: DictConfig,
        
    ):
        generator = Generator.create(
            seed=seed,
            input_sample=input_sample,
            output_dim=generator_output_dim,
            **generator_config,
        )
        
        discriminator_input_sample = generator(input_sample)
        discriminator = Discriminator.create(
            seed=seed,
            input_sample=discriminator_input_sample,
            **discriminator_config,
        )

        return cls(generator=generator, discriminator=discriminator)
    
    def __init__(self, generator: Generator, discriminator: Discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def update(self, gan_inputs: jnp.ndarray, real_batch: jnp.ndarray):
        (
            self.generator,
            self.discriminator,
            info,
            stats_info,
        ) = _update_jit(
            gan_inputs=gan_inputs,
            real_batch=real_batch,
            generator=self.generator,
            discriminator=self.discriminator,
        )
        return info, stats_info

@jax.jit
def _update_jit(
    gan_inputs: jnp.ndarray,
    real_batch: jnp.ndarray,
    generator: Generator,
    discriminator: Discriminator,
):
    new_gen, gen_info, gen_stats_info = generator.update(batch=gan_inputs, discriminator=discriminator)
    new_disc, disc_info, disc_stats_info = discriminator.update(real_batch=real_batch, fake_batch=gen_info.pop("fake_batch"))

    info = {**gen_info, **disc_info}
    stats_info = {**gen_stats_info, **disc_stats_info}
    return (
        new_gen,
        new_disc,
        info,
        stats_info,
    )
