"""Offline lambda-return targets (experiment-design decisions D3 and D4).

For a finished episode S_0, A_0, R_1, S_1, ..., R_T, S_T the lambda-return is
computed backwards:

    G_t = R_{t+1} + gamma * ((1 - lam) * v_{t+1} + lam * G_{t+1})

with the boundary value

    G_T = 0          if the episode terminated (goal reached, etc.)
    G_T = v_T        if it was truncated by the step limit (bootstrap, D4)

``v_{t+1}`` is V(S_{t+1}) for prediction or Q(S_{t+1}, A_{t+1}) for SARSA
control. lam = 0 gives one-step TD targets. lam = 1 gives Monte Carlo returns
on terminated episodes, with a bootstrapped tail on truncated ones.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def lambda_returns(
    rewards: Sequence[float],
    next_values: Sequence[float],
    *,
    terminated: bool,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Return G^lambda_t for t = 0..T-1.

    Args:
        rewards: R_1..R_T (length T).
        next_values: v_1..v_T (length T), the value estimate of the state (or
            state-action) reached after each step. ``next_values[-1]`` is used
            only when the episode was truncated.
        terminated: True if the last transition was a true termination.
    """
    T = len(rewards)
    if len(next_values) != T:
        raise ValueError("rewards and next_values must have the same length")
    if T == 0:
        return np.zeros(0)

    out = np.empty(T)
    g = 0.0 if terminated else float(next_values[-1])
    for t in range(T - 1, -1, -1):
        v_next = g if t == T - 1 else float(next_values[t])
        g = rewards[t] + gamma * ((1.0 - lam) * v_next + lam * g)
        out[t] = g
    return out
