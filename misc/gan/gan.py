from copy import deepcopy
from typing import Tuple

import jax
import jax.numpy as jnp
from omegaconf.dictconfig import DictConfig

from utils import SaveLoadMixin

from .discriminator import Discriminator
from .generator import Generator


class GAN(SaveLoadMixin):
    _save_attrs: Tuple[str] = (
        "generator",
        "discriminator"
    )

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        input_dim: int,
        generator_output_dim: int,
        #
        loss_config: DictConfig,
        generator_config: DictConfig,
        discriminator_config: DictConfig,
    ):
        generator_loss_config = deepcopy(loss_config)
        generator_loss_config["is_generator"] = True
        generator = Generator.create(
            seed=seed,
            input_dim=input_dim,
            output_dim=generator_output_dim,
            loss_config=generator_loss_config,
            **generator_config,
        )

        discriminator_loss_config = deepcopy(loss_config)
        discriminator_loss_config["is_generator"] = False
        discriminator_input_dim = generator(jnp.ones(input_dim, dtype=jnp.float32)).shape
        discriminator = Discriminator.create(
            seed=seed,
            input_dim=discriminator_input_dim,
            loss_config=discriminator_loss_config,
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
    new_disc, disc_info, disc_stats_info = discriminator.update(real_batch=real_batch, fake_batch=gen_info.pop("generations"))

    info = {**gen_info, **disc_info}
    stats_info = {**gen_stats_info, **disc_stats_info}
    return (
        new_gen,
        new_disc,
        info,
        stats_info,
    )
