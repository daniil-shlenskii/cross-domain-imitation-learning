from typing import Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from networks.common import MLP, default_init

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class DeterministicPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, training: bool=False
    ) -> jnp.ndarray:
        features = MLP(
            self.hidden_dims, dropout_rate=self.dropout_rate,
        )(observations, training=training)

        actions = nn.Dense(
            self.action_dim, kernel_init=default_init(),
        )(features)

        return nn.tanh(actions)


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    train_std: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
    ) -> tfd.Distribution:
        features = MLP(
            self.hidden_dims, dropout_rate=self.dropout_rate
        )(observations, training=training)
        means = nn.Dense(
            self.action_dim, kernel_init=default_init()
        )(features)

        if self.train_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init()
            )(features)
        else:
            log_stds = self.param(
                'log_stds', nn.initializers.zeros, (self.action_dim,)
            )
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        normal_dist = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) / temperature
        )
        return tfd.TransformedDistribution(distribution=normal_dist, bijector=tfb.Tanh())