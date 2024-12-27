from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from networks.common import MLP, default_init


class BaseDiscrimiantorAffineTransform(nn.Module):
    n_params: int
    is_params_first: bool
    use_bias: bool

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool=False,
    ):
        x_dim = x.shape[-1]

        if self.is_params_first:
            x_to_process, x_to_zero = jnp.split(x, (self.n_params,), axis=-1)
        else:
            x_to_zero, x_to_process = jnp.split(x, (x_dim - self.n_params,), axis=-1)

        processed_x = nn.Dense(1, kernel_init=default_init(), use_bias=self.use_bias)(x_to_process).squeeze(-1) 
        zeroed_x = (x_to_zero * 0.).mean(-1)
        logits = processed_x + zeroed_x

        return logits

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
            x = MLP(hidden_dims=(*self.hidden_dims, x_dim))(x, train=train)

        state_logits = BaseDiscrimiantorAffineTransform(
            is_params_first=True,
            n_params=n_state_discriminator_params,
            use_bias=self.use_final_layer_bias,
        )(x)
        policy_logits = BaseDiscrimiantorAffineTransform(
            is_params_first=False,
            n_params=x_dim - n_state_discriminator_params,
            use_bias=self.use_final_layer_bias,
        )(x)

        return state_logits, policy_logits
