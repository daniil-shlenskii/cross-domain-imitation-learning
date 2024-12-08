import matplotlib.pyplot as plt
import numpy as np

import wandb
from agents.base_agent import Agent
from utils import apply_model_jit, convert_figure_to_array
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
    learner_state = learner_traj["observations"]
    expert_state = expert_traj["observations"]

    # state and policy logits
    state_learner_logits = apply_model_jit(gail_agent.sample_discriminator, learner_state)
    state_expert_logits = apply_model_jit(gail_agent.sample_discriminator, expert_state)

    # plots
    def logits_to_plot(logits):
        figure = plt.figure(figsize=(5, 5))
        plt.plot(logits, "bo")
        plt.axhline(y=0., color='r', linestyle='-')
        plt.close()
        return figure

    state_learner_figure = logits_to_plot(state_learner_logits)
    state_expert_figure = logits_to_plot(state_expert_logits)

    # priorities hist
    priorities = np.asarray(gail_agent.sample_discriminator.priorities)
    samples = np.random.choice(
        a=len(priorities),
        size=10_000,
        p=priorities,
        replace=True,
    )

    priorities_hist = plt.figure(figsize=(5, 5))
    plt.hist(samples, density=True)
    plt.xlabel("priorities")
    plt.close()

    return state_learner_figure, state_expert_figure, priorities_hist
