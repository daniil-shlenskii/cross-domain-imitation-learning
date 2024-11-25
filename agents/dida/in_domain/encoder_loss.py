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
    for k, v in learner_info.items():
        learner_info[f"learner_{k}"] = v

    expert_loss, expert_info = expert_encoder_loss(
        params=params,
        state=state,
        batch=expert_batch,
        policy_discriminator=policy_discriminator,
        domain_discriminator=domain_discriminator,
        domain_loss_scale=domain_loss_scale,
    )
    for k, v in expert_info.items():
        expert_info[f"expert_{k}"] = v

    loss = (learner_loss + expert_loss) * 0.5
    info = {**learner_info, **expert_info}
    return loss, info