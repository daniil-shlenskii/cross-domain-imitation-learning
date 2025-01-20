import jax
import jax.numpy as jnp

from agents.imitation_learning.cross_domain.domain_encoder.discriminators.base_discriminators import \
    BaseDomainEncoderDiscriminators
from agents.imitation_learning.utils import get_state_pairs
from misc.gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.custom_types import DataType, Params
from utils.math import cosine_similarity_fn


class OrthogonalDiscriminatorLoss:
    def __init__(self, target_reg_scale: float=1., source_reg_scale: float=1.):
        self.target_reg_scale = target_reg_scale
        self.source_reg_scale = source_reg_scale

    def __call__(
        self,
        state_discriminator_params: Params,
        policy_discriminator_params: Params,
        state_discriminator_state: TrainState, 
        policy_discriminator_state: TrainState, 
        target_random_batch: DataType,
        source_random_batch: DataType,
        source_expert_batch: DataType,
    ):
        # downstream losses
        ## state discriminator
        state_discr_downstream_loss, _ = state_discriminator_state.loss_fn(
            state_discriminator_params,
            state=state_discriminator_state,
            fake_batch=target_random_batch["observations"],
            real_batch=source_expert_batch["observations"],
            train=True,
        )

        ## policy discriminator
        target_random_pairs = get_state_pairs(target_random_batch)
        source_random_pairs = get_state_pairs(source_random_batch)
        source_expert_pairs = get_state_pairs(source_expert_batch)
        policy_discr_downstream_loss, _ = policy_discriminator_state.loss_fn(
            policy_discriminator_params,
            state=policy_discriminator_state,
            # fake_batch=jnp.concatenate([target_random_pairs, source_random_pairs]),
            fake_batch=source_random_pairs,
            real_batch=source_expert_pairs,
            train=True,
        )

        # orthogonal regularization
        dim = target_random_batch["observations"].shape[-1]

        ## state grads
        def target_random_state_gen_loss_fn(x: jnp.ndarray):
            logits = state_discriminator_state.apply_fn({"params": state_discriminator_params}, x)
            return state_discriminator_state.loss_fn.generator_loss_fn(logits)
        target_random_state_grad = jax.grad(target_random_state_gen_loss_fn)(target_random_batch["observations"])

        def source_expert_state_gen_loss_fn(x: jnp.ndarray):
            logits = state_discriminator_state.apply_fn({"params": state_discriminator_params}, x)
            return state_discriminator_state.loss_fn.generator_loss_fn(-logits)
        source_expert_state_grad = jax.grad(source_expert_state_gen_loss_fn)(source_expert_batch["observations"])

        ## policy grads
        def target_random_policy_gen_loss_fn(x: jnp.ndarray):
            logits = policy_discriminator_state.apply_fn({"params": policy_discriminator_params}, x)
            return policy_discriminator_state.loss_fn.generator_loss_fn(-logits)
        target_random_policy_grad = jax.grad(target_random_policy_gen_loss_fn)(target_random_pairs)
        target_random_policy_grad = target_random_policy_grad.at[:, :dim].get()

        def source_expert_policy_gen_loss_fn(x: jnp.ndarray):
            logits = policy_discriminator_state.apply_fn({"params": policy_discriminator_params}, x)
            return policy_discriminator_state.loss_fn.generator_loss_fn(logits)
        source_expert_policy_grad = jax.grad(source_expert_policy_gen_loss_fn)(source_expert_pairs)
        source_expert_policy_grad = source_expert_policy_grad.at[:, :dim].get()

        ## regularization
        target_random_cossim = jax.vmap(cosine_similarity_fn)(target_random_state_grad, target_random_policy_grad)
        source_expert_cossim = jax.vmap(cosine_similarity_fn)(source_expert_state_grad, source_expert_policy_grad)

        target_random_cossim_scaled = target_random_cossim * dim**0.5
        source_expert_cossim_scaled = source_expert_cossim * dim**0.5

        cossim_term = (jnp.abs(target_random_cossim_scaled) * self.target_reg_scale + jnp.abs(source_expert_cossim_scaled) * self.source_reg_scale).mean()

        #
        loss = (
            state_discr_downstream_loss +\
            policy_discr_downstream_loss +\
            cossim_term
        )

        return loss, {
            "discriminators/loss": loss,
            "discriminators/cossim_term": cossim_term,
            f"{state_discriminator_state.info_key}_downstream_loss": state_discr_downstream_loss,
            f"{policy_discriminator_state.info_key}_downstream_loss": policy_discr_downstream_loss,
            "discriminators/target_random_cossim": target_random_cossim.mean(),
            "discriminators/target_random_cossim_scaled": target_random_cossim_scaled.mean(),
            "discriminators/source_expert_cossim": source_expert_cossim.mean(),
            "discriminators/source_expert_cossim_scaled": source_expert_cossim_scaled.mean(),
        }
