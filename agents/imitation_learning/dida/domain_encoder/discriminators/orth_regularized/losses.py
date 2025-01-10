import jax
import jax.numpy as jnp

from agents.imitation_learning.dida.domain_encoder.utils import \
    cosine_similarity_fn
from gan.losses import SoftplusLoss
from nn.train_state import TrainState
from utils.types import Params


def apply_fn_wrapper(apply_fn):
    def wrapper(*args, **kwargs):
        logits = apply_fn(*args, **kwargs)
        return logits.mean(), logits
    return wrapper

def _get_orth_regularized_gan_losses(cls):
    class OrthRegularizedPolicyGANLoss(cls):
        def discriminator_loss(
            self,
            params: Params,
            state: TrainState,
            real_batch: jnp.ndarray,
            fake_batch: jnp.ndarray,
            train: bool = True,
        ):
            apply_fn = apply_fn_wrapper(state.apply_fn)
            (_, real_logits), real_grad = jax.value_and_grad(apply_fn, argnums=1, has_aux=True)(
                {"params": params}, real_batch, train=train
            )
            (_, fake_logits), fake_grad = jax.value_and_grad(apply_fn, argnums=1, has_aux=True)(
                {"params": params}, fake_batch, train=train
            )
            loss = self.discriminator_loss_fn(real_logits=real_logits, fake_logits=fake_logits)

            info = {
                f"{state.info_key}_loss": loss,
                "real_logits": real_logits,
                "fake_logits": fake_logits,
                "real_grads": real_grad,
                "fake_grads": fake_grad,
            }
            return loss, info

    class OrthRegularizedStateGANLoss(cls):
        def discriminator_loss(
            self,
            params: Params,
            state: TrainState,
            real_batch: jnp.ndarray,
            fake_batch: jnp.ndarray,
            real_policy_grad: jnp.ndarray,
            fake_policy_grad: jnp.ndarray,
            orth_regularization_scale: float = 1.,
            train: bool = True,
        ):
            apply_fn = apply_fn_wrapper(state.apply_fn)

            # downstream term
            (_, real_logits), real_grad = jax.value_and_grad(apply_fn, argnums=1, has_aux=True)(
                {"params": params}, real_batch, train=train
            )
            (_, fake_logits), fake_grad = jax.value_and_grad(apply_fn, argnums=1, has_aux=True)(
                {"params": params}, fake_batch, train=train
            )
            downstream_loss = self.discriminator_loss_fn(real_logits=real_logits, fake_logits=fake_logits)

            # regularization term
            rr = (cosine_similarity_fn(real_policy_grad, real_grad)**2).mean()
            ff = (cosine_similarity_fn(fake_policy_grad, fake_grad)**2).mean()
            reg = rr + ff

            loss = downstream_loss + reg * orth_regularization_scale

            info = {
                f"{state.info_key}_loss": loss,
                f"{state.info_key}_downstream_loss": downstream_loss,
                f"{state.info_key}_rr": rr,
                f"{state.info_key}_ff": ff,
                "real_logits": real_logits,
                "fake_logits": fake_logits,
            }
            return loss, info
    return OrthRegularizedPolicyGANLoss, OrthRegularizedStateGANLoss

OrthRegularizedPolicySoftplusLoss, OrthRegularizedStateSoftplusLoss = _get_orth_regularized_gan_losses(SoftplusLoss)
