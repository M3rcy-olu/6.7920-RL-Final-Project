"""Action-selection policies."""
from __future__ import annotations

import numpy as np


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """With probability epsilon a uniform random action, else greedy with random tie-breaking."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    best = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best))


def uniform_random(n_actions: int, rng: np.random.Generator) -> int:
    return int(rng.integers(n_actions))
