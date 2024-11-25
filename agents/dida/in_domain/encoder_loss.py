from agents.dida.encoder_loss import expert_encoder_loss, learner_encoder_loss
from gan.discriminator import Discriminator
from nn.train_state import TrainState
from utils.types import DataType, Params


def encoder_loss(
    params: Params,
    state: TrainState,
    batch: DataType,
    expert_batch: DataType,
    policy_discriminator: Discriminator,
    domain_discriminator: Discriminator,
    domain_loss_scale: float,
):
    learner_loss, learner_info = learner_encoder_loss(
        params=params,
        state=state,
        batch=batch,
        policy_discriminator=policy_discriminator,
        domain_discriminator=domain_discriminator,
        domain_loss_scale=domain_loss_scale,
    )
    info_keys = set(learner_info.keys())
    for k in info_keys:
        learner_info[f"learner_{k}"] = learner_info.pop(k)

    expert_loss, expert_info = expert_encoder_loss(
        params=params,
        state=state,
        batch=expert_batch,
        policy_discriminator=policy_discriminator,
        domain_discriminator=domain_discriminator,
        domain_loss_scale=domain_loss_scale,
    )
    for k in info_keys:
        expert_info[f"expert_{k}"] = expert_info.pop(k)

    loss = (learner_loss + expert_loss) * 0.5
    info = {
        f"{state.info_key}_loss": loss,
        **learner_info,
        **expert_info
    }
    return loss, info