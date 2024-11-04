import numpy as np

from utils.types import PRNGKey


def seed_by_key(key: PRNGKey) -> int:
    print(f"{key = }")
    return np.round(np.sum(key)).astype(int)