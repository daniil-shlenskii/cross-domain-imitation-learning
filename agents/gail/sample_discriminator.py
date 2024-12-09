from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

from gan.discriminator import Discriminator
from utils.types import DataType, PRNGKey


class SampleDiscriminator(Discriminator):
    buffer_state_experience: DataType = struct.field(pytree_node=False)
    state_encoder: Callable = struct.field(pytree_node=False)
    sample_size: int = struct.field(pytree_node=False)
    priorities: jnp.ndarray
    ema_decay: float
    temperature: float

    @classmethod
    def create(
        cls,
        *,
        buffer_state_experience: DataType,
        sample_size: int,
        state_encoder: Callable = None,
        ema_decay: float = 0.99,
        temperature: float = 1.,
        **discriminator_kwargs,
    ):
        if state_encoder is None:
            state_encoder = lambda x: x

        exp_size = buffer_state_experience["observations"].shape[0]
        priorities = jnp.ones(exp_size) / float(exp_size)

        return super().create(
            buffer_state_experience=buffer_state_experience,
            state_encoder = state_encoder,
            sample_size=sample_size,
            priorities=priorities,
            ema_decay=ema_decay,
            temperature=temperature,
            info_key="sample_discriminator",
            _save_attrs = ("state", "priorities"),
            **discriminator_kwargs,
        )

    def update(self, *, expert_batch: DataType, learner_batch: DataType, state_encoder: Callable=None):
        new_sample_discr = self
        if state_encoder is not None:
            new_sample_discr = self.replace(state_encoder=state_encoder)
        new_sample_discr, info, stats_info = _update(
            sample_discriminator=new_sample_discr,
            learner_batch=learner_batch,
            expert_batch=expert_batch,
        )
        return new_sample_discr, info, stats_info

    @jax.jit
    def sample(self, rng: PRNGKey):
        new_rng, key = jax.random.split(rng)

        batch_idcs = jax.random.choice(
            key=key,
            a=self.priorities.shape[0],
            shape=(self.sample_size,),
            p=self.priorities,
            replace=False,
        )

        batch = {}
        for k, v in self.buffer_state_experience.items():
            batch[k] = v.at[batch_idcs].get()

        return new_rng, batch

    @jax.jit
    def _get_priorities(self):
        states = self.state_encoder(self.buffer_state_experience["observations"])
        logits = self(states)
        shifted_logits = logits - logits.min()
        normalized_logits =  shifted_logits / shifted_logits.max()

        relevance = 1. - normalized_logits
        priorities = jax.nn.softmax(relevance / self.temperature)
        return priorities


@jax.jit
def _update(
    sample_discriminator: SampleDiscriminator,
    learner_batch: DataType,
    expert_batch: DataType,
):
    # update discriminator
    new_sample_discr, info, stats_info = Discriminator.update(
        self=sample_discriminator,
        real_batch=expert_batch["observations"],
        fake_batch=learner_batch["observations"],
    )

    # update priorities
    priorities = new_sample_discr._get_priorities()
    new_priorities = sample_discriminator.priorities * sample_discriminator.ema_decay + priorities * (1 - sample_discriminator.ema_decay)
    new_sample_discr = new_sample_discr.replace(priorities=new_priorities)

    return new_sample_discr, info, stats_info
