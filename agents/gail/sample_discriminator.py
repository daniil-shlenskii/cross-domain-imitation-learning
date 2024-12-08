import functools

import jax
import jax.numpy as jnp
from omegaconf.dictconfig import DictConfig

from gan.discriminator import Discriminator
from utils.types import BufferState, DataType, PRNGKey


class SampleDisciminator(Discriminator):
    buffer_state_experience: DataType
    priorities: jnp.ndarray
    update_priorities_every: int

    @classmethod
    def create(
        cls,
        *,
        buffer_state: BufferState,
        update_priorities_every: int = 1,
        **discriminator_kwargs: DictConfig, 
    ):
        buffer_state_experience = buffer_state.experience

        exp_size = buffer_state_experience["observations"].shape[0]
        priorities = jnp.ones(exp_size) / float(exp_size)

        return super().create(
            buffer_state_experience=buffer_state_experience,
            priorities=priorities,
            update_priorities_every=update_priorities_every,
            **discriminator_kwargs,
        )

    def update(self, *, expert_batch: DataType, learner_batch: DataType):
        new_sample_discriminator, info, stats_info = _update(
            sample_discriminator=self,
            learner_batch=learner_batch,
            expert_batch=expert_batch,
        )
        return new_sample_discriminator, info, stats_info

    @functools(jax.jit, static_argnames="sample_size")
    def sample(self, rng: PRNGKey, sample_size: int):
        new_rng, key = jax.random_split(rng)

        batch_idcs = jax.random.choice(
            key=key,
            a=self.buffer_state_experience["observations"].shape[0],
            shape=sample_size,
            p=self.priorities,
            replace=False,
        )

        batch = jax.tree.map(
            lambda x: x.at[batch_idcs].get(),
            self.buffer_state_experience,
            is_leaf=lambda x: isinstance(x, jnp.ndarray)
        )

        return new_rng, batch

    @jax.jit
    def get_priorities(self, expert_batch: DataType):
        states = expert_batch["observations"]
        logits = self(states)
        shifted_logits = logits - logits.min()
        normalized_logits =  shifted_logits / shifted_logits.max()

        relevance = 1 - normalized_logits
        priorities = relevance / relevance.sum()
        return priorities


@functools.partial(jax.jit, static_arganames="update_priorities")
def _update(
    sample_discriminator: SampleDisciminator,
    learner_batch: DataType,
    expert_batch: DataType,
    update_priorities: bool
):
    # update discriminator
    new_sample_discr, info, stats_info = Discriminator.update(
        self=sample_discriminator,
        real_batch=expert_batch,
        fake_batch=learner_batch,
    )

    # update priorities
    if update_priorities:
        priorities = sample_discriminator.get_priorities(sample_discriminator.buffer_state_experience)
        new_priorities = (priorities + sample_discriminator.priorities) * 0.5
        new_sample_discr = new_sample_discr.replace(priorities=new_priorities)

    return new_sample_discr, info, stats_info
