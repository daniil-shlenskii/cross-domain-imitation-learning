import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from misc.gan.discriminator import Discriminator
from utils import sample_batch_jit
from utils.custom_types import DataType

from .gwil_enot import GWILENOT


class GWILENOTwGAIL(GWILENOT):
    policy_discriminator: Discriminator

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        policy_discriminator_config: DictConfig,
        use_pairs: bool = True,
        **kwargs,
    ):
        assert use_pairs

        gwil_enot = super().create(
            seed=seed,
            use_pairs=use_pairs,
            policy_discriminator=None,
            **kwargs,
        )
        target_dim = gwil_enot.ot_target_buffer_state.experience["observations"].shape[-1]
        policy_discriminator = instantiate(
            policy_discriminator_config,
            seed=seed,
            input_dim=target_dim * 2,
            _recursive_=False,
        )

        return gwil_enot.replace(policy_discriminator=policy_discriminator)

    @jax.jit
    def update(self, *, target_expert_batch: DataType): 
        ot_source_batch = target_expert_batch

        # sample ot target batch
        new_rng, ot_target_batch = sample_batch_jit(
            self.rng, self.ot_buffer, self.ot_target_buffer_state
        )

        # sample source expert batch
        new_rng, source_expert_batch = sample_batch_jit(
            new_rng, self.source_expert_buffer, self.source_expert_buffer_state
        )

        # process dict batch
        ot_source_batch = self.process_dict_batch_fn(ot_source_batch)
        ot_target_batch = self.process_dict_batch_fn(ot_target_batch)
        target_expert_batch = self.process_dict_batch_fn(target_expert_batch)
        source_expert_batch = self.process_dict_batch_fn(source_expert_batch)

        # update enot
        new_enot, enot_info, enot_stats_info = self.enot.update(
            source=ot_source_batch, target=ot_target_batch,
        )

        # update policy discriminator
        target_hat_expert_batch = new_enot(target_expert_batch)
        new_policy_discr, policy_discr_info, policy_discr_stats_info = self.policy_discriminator.update(
            real_batch=source_expert_batch,
            fake_batch=target_hat_expert_batch,
        )

        # update reward transform
        base_rewards = self.get_base_rewards(target_expert_batch)
        new_reward_transform, reward_transform_info = self.reward_transform.update(base_rewards)

        self = self.replace(
            rng=new_rng,
            enot=new_enot,
            reward_transform=new_reward_transform,
            policy_discriminator=new_policy_discr,
        )
        info = {**enot_info, **reward_transform_info, **policy_discr_info}
        stats_info = {**enot_stats_info, **policy_discr_stats_info}
        return self, info, stats_info

    @jax.jit
    def get_base_rewards(self, target_expert_batch: jnp.ndarray) -> jnp.ndarray:
        target_hat_expert_batch = self.enot(target_expert_batch)
        return self.policy_discriminator(target_hat_expert_batch)
