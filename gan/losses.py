import jax
import jax.numpy as jnp
from gan.base_losses import d_softplus_loss, gradient_penalty
from nn.train_state import TrainState
from utils.types import Params, PRNGKey


def d_softplus_loss_with_gradient_penalty(
    params: Params,
    state: TrainState,
    real_batch: jnp.ndarray,
    fake_batch: jnp.ndarray,
    gradient_penalty_coef: float,
    key: PRNGKey,
):
    real_logits = state.apply_fn({"params": params}, real_batch, train=True)
    fake_logits = state.apply_fn({"params": params}, fake_batch, train=True)
    loss = d_softplus_loss(real_logits=real_logits, fake_logits=fake_logits)

    disc_grad_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x, train=True))
    penalty = gradient_penalty(key=key, real_batch=real_batch, fake_batch=fake_batch, discriminator_grad_fn=disc_grad_fn)

    info = {
        f"{state.info_key}_loss": loss,
        f"{state.info_key}_gradient_penalty": penalty,
        "real_logits": real_logits,
        "fake_logits": fake_logits,
    }
    return loss + penalty * gradient_penalty_coef, info