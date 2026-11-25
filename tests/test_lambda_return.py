"""Tests for the offline lambda-return targets (decision D3, boundary rule D4)."""
from __future__ import annotations

import numpy as np
import pytest

from td.lambda_return import lambda_returns

GAMMA = 0.9


def mc_returns(rewards, gamma, tail=0.0):
    out, g = [], tail
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def forward_view(rewards, next_values, bootstrap, gamma, lam):
    """Textbook definition: (1-lam) * sum lam^(n-1) G_{t:t+n} + lam^(T-t-1) G_t."""
    T = len(rewards)
    values = list(next_values[:-1]) + [bootstrap]  # V(S_1) .. V(S_T)
    out = []
    for t in range(T):
        def n_step(n):
            g = sum(gamma**k * rewards[t + k] for k in range(n))
            return g + gamma**n * values[t + n - 1]
        full = n_step(T - t)
        g_lam = (1 - lam) * sum(lam ** (n - 1) * n_step(n) for n in range(1, T - t)) + lam ** (T - t - 1) * full
        out.append(g_lam)
    return out


def test_lambda_zero_is_one_step_td_target():
    rewards = [0.0, 1.0, 2.0]
    next_values = [5.0, 6.0, 7.0]
    g = lambda_returns(rewards, next_values, terminated=False, gamma=GAMMA, lam=0.0)
    np.testing.assert_allclose(g, [0 + GAMMA * 5, 1 + GAMMA * 6, 2 + GAMMA * 7])


def test_lambda_one_on_terminated_episode_is_monte_carlo_return():
    rewards = [0.0, 0.0, 1.0]
    next_values = [5.0, 6.0, 7.0]  # must be ignored entirely
    g = lambda_returns(rewards, next_values, terminated=True, gamma=GAMMA, lam=1.0)
    np.testing.assert_allclose(g, mc_returns(rewards, GAMMA))


def test_terminated_episode_never_bootstraps_from_final_value():
    g = lambda_returns([1.0], [100.0], terminated=True, gamma=GAMMA, lam=0.0)
    np.testing.assert_allclose(g, [1.0])


def test_truncated_episode_bootstraps_from_final_value_at_lambda_one():
    rewards = [0.0, 1.0]
    next_values = [3.0, 10.0]  # V(S_T) = 10 is the bootstrap
    g = lambda_returns(rewards, next_values, terminated=False, gamma=GAMMA, lam=1.0)
    np.testing.assert_allclose(g, mc_returns(rewards, GAMMA, tail=10.0))


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.8, 1.0])
@pytest.mark.parametrize("terminated", [True, False])
def test_recursion_matches_textbook_forward_view(lam, terminated):
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=6).tolist()
    next_values = rng.normal(size=6).tolist()
    boot = 0.0 if terminated else next_values[-1]
    g = lambda_returns(rewards, next_values, terminated=terminated, gamma=GAMMA, lam=lam)
    np.testing.assert_allclose(g, forward_view(rewards, next_values, boot, GAMMA, lam))
