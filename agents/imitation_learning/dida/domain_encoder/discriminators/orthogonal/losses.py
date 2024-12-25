import jax.numpy as jnp

from gan.losses import GANLoss, SoftplusLoss
from nn.train_state import TrainState
from utils.types import Params


def _orthogonal_discrimiantors_gan_loss_decorator(cls: GANLoss):
    class DiscrimantorsGANLoss(cls):
        def __init__(self, *args, **kwargs):
            is_generator = kwargs.pop("is_generator", False)
            super().__init__(*args, is_generator=is_generator, **kwargs)

        def __call__(
            self,
            params: Params,
            state: TrainState,
            *,
            target_random_pairs: jnp.ndarray,
            source_random_pairs: jnp.ndarray,
            source_expert_pairs: jnp.ndarray,
        ):
            # get logits
            target_random_state_logits, target_random_policy_logits = state.apply_fn(
                {"params": params}, target_random_pairs,
            )
            source_random_state_logits, source_random_policy_logits = state.apply_fn(
                {"params": params}, source_random_pairs,
            )
            source_expert_state_logits, source_expert_policy_logits = state.apply_fn(
                {"params": params}, source_expert_pairs,
            )

            # get state loss
            state_loss = self.discriminator_loss_fn(
                real_logits=jnp.concatenate([source_random_state_logits, source_expert_state_logits]),
                fake_logits=target_random_state_logits,
            )

            # get policy loss
            policy_loss = self.discriminator_loss_fn(
                real_logits=source_expert_policy_logits,
                fake_logits=jnp.concatenate([target_random_policy_logits, source_random_policy_logits]),
            )

            # loss
            loss = (state_loss + policy_loss) * 0.5

            info = {
                f"{state.info_key}/loss": loss,
                f"{state.info_key}/state_loss": state_loss,
                f"{state.info_key}/policy_loss": policy_loss,
            }
            return loss, info
    return DiscrimantorsGANLoss

DiscriminatorsSoftplusLoss = _orthogonal_discrimiantors_gan_loss_decorator(SoftplusLoss)
