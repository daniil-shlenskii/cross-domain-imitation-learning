from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant

from networks.common import MLP, default_init


class BaseDiscrimiantorAffineTransform(nn.Module):
    is_params_first: bool
    n_params: int
    use_bias: bool

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False,
    ):
        x_dim = x.shape[-1]

        if self.is_params_first:
            x_to_process, x_to_zero = jnp.split(x, (self.n_params, x_dim - self.n_params), axis=-1)
        else:
            x_to_zero, x_to_process = jnp.split(x, (x_dim - self.n_parms, self.n_params), axis=-1)

        processed_x = nn.Dense(self.n_params, kernel_init=default_init(), use_bias=self.use_bias)(x_to_process)
        zeroed_x = x_to_zero * 0.

        if self.is_params_first:
            result = jnp.concatenate([processed_x, zeroed_x], axis=-1)
        else:
            result = jnp.concatenate([zeroed_x, processed_x], axis=-1)
        return result

class DiscriminatorsModule(nn.Module):
    n_state_discriminator_params: int = None
    hidden_dims: Sequence[int] = None
    use_final_layer_bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False,
    ):
        x_dim = x.shape[-1]

        if self.n_state_discriminator_params is None:
            n_state_discriminator_params = x_dim // 2
        else:
            n_state_discriminator_params = self.n_state_discriminator_params

        if self.hidden_dims is not None:
            print(f"{x.shape = }")
            x = MLP(hidden_dims=(*self.hidden_dims, x_dim))(x, train=train)

        state_logits = BaseDiscrimiantorAffineTransform(
            is_params_first=True,
            n_params=n_state_discriminator_params,
            use_bias=self.use_final_layer_bias,
        )
        policy_logits = BaseDiscrimiantorAffineTransform(
            is_params_first=False,
            n_params=x_dim - n_state_discriminator_params,
            use_bias=self.use_final_layer_bias,
        )

        return state_logits, policy_logits
