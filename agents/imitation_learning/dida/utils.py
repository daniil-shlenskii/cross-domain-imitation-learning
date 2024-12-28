import matplotlib.pyplot as plt
import numpy as np

from agents.base_agent import Agent
from agents.imitation_learning.utils import get_state_pairs
from utils import apply_model_jit
from utils.types import DataType


def get_discriminators_logits_plots(
    dida_agent: Agent,
    learner_trajs: DataType,
):
    observation_keys = ["observations", "observations_next"]

    # get trajectories
    end_of_firt_traj_idx = np.argmax(learner_trajs["dones"])
    learner_traj = {k: learner_trajs[k][:end_of_firt_traj_idx] for k in observation_keys}
    expert_traj = {k: dida_agent.expert_buffer_state.experience[k][0, :end_of_firt_traj_idx] for k in observation_keys}

    for k in observation_keys:
        learner_traj[k] = dida_agent._preprocess_observations(learner_traj[k])
        expert_traj[k] = dida_agent._preprocess_expert_observations(expert_traj[k])

    # get logits
    discs = dida_agent.domain_encoder.discriminators
    target_state_pairs = get_state_pairs(learner_traj)
    source_state_pairs = get_state_pairs(expert_traj)

    ## state logits
    if discs.has_state_discriminator_paired_input:
        target_state_logits = apply_model_jit(discs.state_discriminator, target_state_pairs)
        source_state_logits = apply_model_jit(discs.state_discriminator, source_state_pairs)
    else:
        target_state_logits = apply_model_jit(discs.state_discriminator, learner_traj["observations"])
        source_state_logits = apply_model_jit(discs.state_discriminator, expert_traj["observations"])

    ## policy logits
    target_policy_logits = apply_model_jit(discs.policy_discriminator, target_state_pairs)
    source_policy_logits = apply_model_jit(discs.policy_discriminator, source_state_pairs)

    # plots
    def logits_to_plot(logits):
        figure = plt.figure(figsize=(5, 5))
        plt.plot(logits, "bo")
        plt.axhline(y=0., color='r', linestyle='-')
        plt.close()
        return figure

    ## state plots
    target_state_plot = logits_to_plot(target_state_logits)
    source_state_plot = logits_to_plot(source_state_logits)

    ## policy plots
    target_policy_plot = logits_to_plot(target_policy_logits)
    source_policy_plot = logits_to_plot(source_policy_logits)

    return target_state_plot, source_state_plot, target_policy_plot, source_policy_plot
