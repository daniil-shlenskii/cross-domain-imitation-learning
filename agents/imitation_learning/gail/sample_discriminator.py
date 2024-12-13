import jax
import jax.numpy as jnp
from flax import struct
from gan.discriminator import Discriminator
from utils.types import DataType, PRNGKey


class SampleDiscriminator(Discriminator):
    buffer_state_experience: DataType = struct.field(pytree_node=False)
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
        ema_decay: float = 0.99,
        temperature: float = 1.,
        **discriminator_kwargs,
    ):
        exp_size = buffer_state_experience["observations"].shape[0]
        priorities = jnp.ones(exp_size) / float(exp_size)

        return super().create(
            buffer_state_experience=buffer_state_experience,
            sample_size=sample_size,
            priorities=priorities,
            ema_decay=ema_decay,
            temperature=temperature,
            info_key="sample_discriminator",
            _save_attrs = ("state", "priorities"),
            **discriminator_kwargs,
        )

    def update(self, *, expert_batch: DataType, learner_batch: DataType, expert_encoder=None):
        new_sample_discr, info, stats_info = _update(
            sample_discriminator=self,
            learner_batch=learner_batch,
            expert_batch=expert_batch,
            expert_encoder=expert_encoder,
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
    def get_priorities(self, observation_pairs, expert_encoder=None):
        if expert_encoder is not None:
            observations, observations_next = jnp.split(observation_pairs, 2, axis=1)
            observations = expert_encoder(observations)
            observations_next = expert_encoder(observations_next)
            observation_pairs = self._make_pairs(observations, observations_next)
        logits = self(observation_pairs)
        shifted_logits = logits - logits.min()
        normalized_logits =  shifted_logits / shifted_logits.max()

        relevance = 1. - normalized_logits
        priorities = jax.nn.softmax(relevance / self.temperature)
        return priorities

    def _make_pairs(self, observations: jnp.ndarray, observations_next: jnp.ndarray):
        return jnp.concatenate([
            observations, observations_next
        ], axis=1)


@jax.jit
def _update(
    sample_discriminator: SampleDiscriminator,
    learner_batch: DataType,
    expert_batch: DataType,
    expert_encoder,
):
    # prepare observation pairs
    learner_pairs = sample_discriminator._make_pairs(
        learner_batch["observations"], learner_batch["observations_next"]
    )
    expert_pairs = sample_discriminator._make_pairs(
        expert_batch["observations"], expert_batch["observations_next"]
    )

    # update discriminator
    new_sample_discr, info, stats_info = Discriminator.update(
        self=sample_discriminator,
        real_batch=expert_pairs,
        fake_batch=learner_pairs,
    )

    # update priorities
    exp_pairs = jnp.concatenate([
        sample_discriminator.buffer_state_experience["observations"],
        sample_discriminator.buffer_state_experience["observations_next"],
    ], axis=1)
    priorities = new_sample_discr.get_priorities(exp_pairs, expert_encoder)
    new_priorities = sample_discriminator.priorities * sample_discriminator.ema_decay + priorities * (1 - sample_discriminator.ema_decay)
    new_sample_discr = new_sample_discr.replace(priorities=new_priorities)

    return new_sample_discr, info, stats_info
