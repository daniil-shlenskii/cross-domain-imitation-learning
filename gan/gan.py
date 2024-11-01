from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from flax.struct import PyTreeNode

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from nn.train_state import TrainState
from utils.utils import instantiate_optimizer
from utils.types import Params, PRNGKey

from gan.discriminator import Discriminator
from gan.generator import Generator


class GAN(PyTreeNode):
    discriminator: Discriminator
    generator: Generator

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        input_sample: jnp.ndarray,
        generator_output_dim: int,
        #
        generator_module_config: DictConfig,
        discriminator_module_config: DictConfig,
        #
        generator_optimizer_config: DictConfig,
        discriminator_optimizer_config: DictConfig,
    ):
        generator = Generator.create(
            seed=seed,
            input_sample=input_sample,
            output_dim=generator_output_dim,
            module_config=generator_module_config,
            optimizer_config=generator_optimizer_config,
        )
        
        discriminator_input_sample = generator(input_sample)
        discriminator = discriminator.create(
            seed=seed,
            input_sample=discriminator_input_sample,
            module_config=discriminator_module_config,
            optimizer_config=discriminator_optimizer_config,
        )

        return cls(generator=generator, discriminator=discriminator)

    def update(self, batch: jnp.ndarray):
        (
            self.generator,
            self.discriminator,
            info,
            stats_info,
        ) = _update_jit(
            batch,
            generator=self.generator,
            discriminator=self.discriminator
        )
        return info, stats_info
    
@jax.jit
def _update_jit(
    batch: jnp.ndarray,
    generator: Generator,
    discriminator: Discriminator,
):
    new_gen, gen_info, gen_stats_info = generator.update(batch=batch, discriminator=discriminator)
    new_disc, disc_info, disc_stats_info = discriminator.update(real_batch=batch, fake_batch=gen_info.pop("fake_batch"))

    info = {**gen_info, **disc_info}
    stats_info = {**gen_stats_info, **disc_stats_info}
    return (
        new_gen,
        new_disc,
        info,
        stats_info,
    )
