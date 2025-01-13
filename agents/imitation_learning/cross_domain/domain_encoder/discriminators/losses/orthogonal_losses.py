import jax
import jax.numpy as jnp
from gan.losses import SoftplusLoss

from nn.train_state import TrainState
from utils import cosine_similarity_fn
from utils.custom_types import Params


def _get_orthogonal_losses(cls):
    def _apply_fn_wrapper(apply_fn):
        def wrapper(*args, **kwargs):
            logits = apply_fn(*args, **kwargs)
            return logits.mean(), logits
        return wrapper

    class OrthogonalPolicyDiscriminatorLoss(cls):
        def discriminator_loss(
            self,
            params: Params,
            state: TrainState,
            target_random_state_pairs: jnp.ndarray,
            source_random_state_pairs: jnp.ndarray,
            source_expert_state_pairs: jnp.ndarray,
            train: bool = True,
        ):
            _apply_fn = _apply_fn_wrapper(state.apply_fn)

            (_, source_expert_state_pairs_logits), source_expert_state_pairs_grad = jax.value_and_grad(_apply_fn, argnums=1, has_aux=True)(
                {"params": params}, source_expert_state_pairs, train=train
            )
            (_, target_random_state_pairs_logits), target_random_state_pairs_grad = jax.value_and_grad(_apply_fn, argnums=1, has_aux=True)(
                {"params": params}, target_random_state_pairs, train=train
            )
            source_random_state_pairs_logits = jax.value_and_grad(state.apply_fn, argnums=1, has_aux=True)(
                {"params": params}, source_random_state_pairs, train=train
            )
            loss = self.discriminator_loss_fn(
                real_logits=source_expert_state_pairs_logits,
                fake_logits=jnp.concatenate([target_random_state_pairs_logits, source_random_state_pairs_logits])
            )

            info = {
                f"{state.info_key}_loss": loss,
                "target_random_state_pairs_grad": target_random_state_pairs_grad,
                "source_expert_state_pairs_grad": source_expert_state_pairs_grad,
            }
            return loss, info

    class OrthogonalStateDiscriminatorLoss(cls):
        def discriminator_loss(
            self,
            params: Params,
            state: TrainState,
            target_random_states: jnp.ndarray,
            source_expert_states: jnp.ndarray,
            target_random_state_policy_grad: jnp.ndarray,
            source_expert_state_policy_grad: jnp.ndarray,
            reg_scale: float = 1.,
            train: bool = True,
        ):
            _apply_fn = _apply_fn_wrapper(state.apply_fn)

            # downstream loss
            (_, target_random_states_logits), target_random_states_grad = jax.value_and_grad(_apply_fn, argnums=1, has_aux=True)(
                {"params": params}, target_random_states, train=train
            )
            (_, source_expert_states_logits), source_expert_states_grad = jax.value_and_grad(_apply_fn, argnums=1, has_aux=True)(
                {"params": params}, source_expert_states, train=train
            )
            downstream_loss = self.discriminator_loss_fn(
                fake_logits=target_random_states_logits,
                real_logits=source_expert_states_logits,
            )

            # regularization term
            dim = target_random_states.shape[-1]
            ff = (cosine_similarity_fn(target_random_states_grad, target_random_state_policy_grad)**2).mean() * dim**0.5
            rr = (cosine_similarity_fn(source_expert_states_grad, source_expert_state_policy_grad)**2).mean() * dim**0.5
            reg_loss = rr + ff

            loss = downstream_loss + reg_loss * reg_scale

            info = {
                f"{state.info_key}_loss": loss,
                f"{state.info_key}_downstream_loss": downstream_loss,
                f"{state.info_key}_reg_loss": reg_loss,
                f"{state.info_key}_rr": rr,
                f"{state.info_key}_ff": ff,
            }
            return loss, info
    return OrthogonalStateDiscriminatorLoss, OrthogonalPolicyDiscriminatorLoss

OrthogonalStateDiscriminatorSoftplusLoss, OrthogonalPolicyDiscriminatorSoftplusLoss = _get_orthogonal_losses(SoftplusLoss)
