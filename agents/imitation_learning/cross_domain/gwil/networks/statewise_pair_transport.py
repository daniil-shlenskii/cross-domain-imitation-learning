import flax.linen as nn
import jax.numpy as jnp

from nn.networks import networks_mapping


class StatewisePairTransort(nn.Module):
    model_type: str
    n_blocks: int
    hidden_dim: int
    out_dim: int

    def setup(self):
        self.state_map = networks_mapping[self.model_type](
            n_blocks=self.n_blocks, hidden_dim=self.hidden_dim, out_dim=self.out_dim//2
        )

    def __call__(self, state_pairs: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        states1, states2 = jnp.split(state_pairs, 2, axis=-1)
        states_mapped1 = self.state_map(states1, train=train)
        states_mapped2 = self.state_map(states2, train=train)
        state_pairs_mapped = jnp.concatenate([states_mapped1, states_mapped2], axis=-1)
        return state_pairs_mapped
