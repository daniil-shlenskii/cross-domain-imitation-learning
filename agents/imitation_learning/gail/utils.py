import matplotlib.pyplot as plt
import numpy as np

from agents.base_agent import Agent
from agents.imitation_learning.utils import get_state_pairs
from utils import apply_model_jit
from utils.types import DataType


def get_policy_discriminator_logits_plots(
    gail_agent: Agent,
    learner_trajs: DataType,
):
    observation_keys = ["observations", "observations_next"]

    # get trajectories
    end_of_firt_traj_idx = np.argmax(learner_trajs["dones"])
    learner_traj = {k: learner_trajs[k][:end_of_firt_traj_idx] for k in observation_keys}
    expert_traj = {k: gail_agent.expert_buffer_state.experience[k][0, :end_of_firt_traj_idx] for k in observation_keys}

    # preprocess trjectories
    for k in observation_keys:
        learner_traj[k] = gail_agent._preprocess_observations(learner_traj[k])
        expert_traj[k] = gail_agent._preprocess_expert_observations(expert_traj[k])

    # get state pairs 
    learner_state_pairs_embeddings = get_state_pairs(learner_traj)
    expert_state_pairs_embeddings = get_state_pairs(expert_traj)

    # get state pairs logits
    policy_discriminator = gail_agent.policy_discriminator
    policy_learner_logits = apply_model_jit(policy_discriminator, learner_state_pairs_embeddings)
    policy_expert_logits = apply_model_jit(policy_discriminator, expert_state_pairs_embeddings)

    # plots
    def logits_to_plot(logits):
        figure = plt.figure(figsize=(5, 5))
        plt.plot(logits, "bo")
        plt.axhline(y=0., color='r', linestyle='-')
        plt.close()
        return figure

    policy_learner_figure = logits_to_plot(policy_learner_logits)
    policy_expert_figure = logits_to_plot(policy_expert_logits)

    return policy_learner_figure, policy_expert_figure
