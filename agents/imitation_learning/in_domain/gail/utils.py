from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

from misc.gan import Discriminator
from utils import apply_model_jit

DISCRIMINATOR_REAL_LABELS = {
    "state": {"SR", "SE"},
    "policy": {"SE"},
}

def get_discriminator_logits_plot(logits: np.ndarray):
    figure = plt.figure(figsize=(5, 5))
    plt.plot(logits, "bo")
    plt.axhline(y=0., color='r', linestyle='-')
    plt.close()
    return figure

def get_trajs_discriminator_logits_and_accuracy(
    discriminator: Discriminator,
    traj_dict: dict,
    keys_to_use: Sequence,
    discriminator_key: Literal["policy", "state"],
):
    info_key_prefix = f"{discriminator.state.info_key}"
    # get_logits
    logits_dict = {
        k: apply_model_jit(discriminator, traj_dict[k]) for k in keys_to_use
    }

    # get accuracy
    accuracy_dict = {}
    for k, logits in logits_dict.items():
        if k in DISCRIMINATOR_REAL_LABELS[discriminator_key]:
            accuracy = (logits > 0).mean()
        else:
            accuracy = (logits < 0).mean()
        accuracy_dict[f"{info_key_prefix}/{k}_accuracy"] = accuracy

    # plots
    figure_dict = {}
    for k, logits in logits_dict.items():
        figure = get_discriminator_logits_plot(logits)
        figure_dict[f"{info_key_prefix}/{k}_logits"] = figure

    return accuracy_dict, figure_dict
