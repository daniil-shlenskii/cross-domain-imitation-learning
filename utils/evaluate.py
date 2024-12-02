from typing import Dict

import gymnasium as gym
import jax
import numpy as np


@jax.jit
def apply_model_jit(model, x):
    return model(x)

