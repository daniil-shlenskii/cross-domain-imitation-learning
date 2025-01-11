from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)
