import matplotlib.pyplot as plt
import numpy as np

from agents.base_agent import Agent
from utils import apply_model_jit
from utils.types import DataType

MIN_TRAJECTORY_SIZE = 100

def get_sample_discriminator_hists(
    gail_agent: Agent,
    learner_trajs: DataType,
):
    observation_keys = ["observations", "observations_next"]

    # get trajectories
    end_of_firt_traj_idx = np.argmax(learner_trajs["dones"])
    end_of_firt_traj_idx = max(end_of_firt_traj_idx, MIN_TRAJECTORY_SIZE)
    learner_traj = {k: learner_trajs[k][:end_of_firt_traj_idx] for k in observation_keys}
    expert_traj = {k: gail_agent.expert_buffer_state.experience[k][0, :end_of_firt_traj_idx] for k in observation_keys}

    # state and policy embeddings
    learner_states = gail_agent._preprocess_observations(learner_traj["observations"])
    expert_states = gail_agent._preprocess_expert_observations(expert_traj["observations"])

    # state and policy logits
    state_learner_logits = apply_model_jit(gail_agent.sample_discriminator, learner_states)
    state_expert_logits = apply_model_jit(gail_agent.sample_discriminator, expert_states)

    # plots
    figsize = (5, 5)
    def logits_to_plot(logits):
        figure = plt.figure(figsize=figsize)
        plt.plot(logits, "bo")
        plt.axhline(y=0., color='r', linestyle='-')
        plt.close()
        return figure

    state_learner_figure = logits_to_plot(state_learner_logits)
    state_expert_figure = logits_to_plot(state_expert_logits)

    # priorities hist
    def priorities_to_plot(priorities):
        figure = plt.figure(figsize=figsize)
        plt.plot(priorities, "go")
        plt.axhline(y=1/len(priorities), color='r', linestyle='-')
        plt.close()
        return figure

    priorities = gail_agent.sample_discriminator.get_priorities(expert_states)
    priorities_acc = gail_agent.sample_discriminator.priorities[:end_of_firt_traj_idx]

    priorities_hist = priorities_to_plot(priorities)
    priorities_acc_hist = priorities_to_plot(priorities_acc)

    return state_learner_figure, state_expert_figure, priorities_hist, priorities_acc_hist
