import jax.numpy as jnp
from flax import linen as nn

from nn.networks import networks_mapping


class Critic(nn.Module):
    model_type: str
    n_blocks: int
    hidden_dim: int
    out_dim: int = None

    def setup(self):
        self.critic_net = networks_mapping[self.model_type](
            n_blocks=self.n_blocks, hidden_dim=self.hidden_dim, out_dim=1, squeeze=True
        )
    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        input_array = jnp.concatenate([observations, actions], axis=-1)
        values = self.critic_net(input_array, train=train)
        return values
