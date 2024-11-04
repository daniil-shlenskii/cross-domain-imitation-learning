from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from networks.common import MLP, default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


def _rescale_from_tanh(x: jnp.ndarray, low: float, high: float) -> jnp.ndarray:
    x = (x + 1) / 2 # (-1, 1) -> (0, 1)
    return x * (high - low) + low


class DeterministicPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, train: bool=False
    ) -> jnp.ndarray:
        features = MLP(
            self.hidden_dims, dropout_rate=self.dropout_rate, activate_final=True,
        )(observations, train=train)
        features = nn.Dense(
            self.action_dim, kernel_init=default_init(),
        )(features)
        actions = nn.tanh(features)
        if self.low is None or self.high is None:
            return actions
        return _rescale_from_tanh(actions)

class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        train: bool = False,
    ) -> distrax.Distribution:
        features = MLP(
            self.hidden_dims, dropout_rate=self.dropout_rate, activate_final=True,
        )(observations, train=train)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(features)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(features)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        return TanhMultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high
        )

class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: Optional[jnp.ndarray] = None,
        high: Optional[jnp.ndarray] = None,
    ):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):
            def _forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    _rescale_from_tanh,
                    forward_log_det_jacobian=_forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1,
                )
            )

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())