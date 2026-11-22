"""Tests for the true online TD(lambda) / Sarsa(lambda) update rule (D3 proposal)."""
from __future__ import annotations

import numpy as np
import pytest

from td.learner import Episode, LinearTD

GAMMA = 0.9
# A trajectory that revisits states, so online and offline rules differ.
STATES = [0, 1, 0, 2, 1, 3]
REWARDS = [0.0, 1.0, 0.0, 0.5, 2.0]


def classic_online_td0(states, rewards, terminated, alpha, gamma, n, v0=None):
    v = np.zeros(n) if v0 is None else v0.copy()
    for t, r in enumerate(rewards):
        last = t == len(rewards) - 1
        v_next = 0.0 if (last and terminated) else v[states[t + 1]]
        v[states[t]] += alpha * (r + gamma * v_next - v[states[t]])
    return v


def sequential_mc(states, rewards, alpha, gamma, n, v0=None):
    v = np.zeros(n) if v0 is None else v0.copy()
    g, returns = 0.0, []
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    for s, g in zip(states, reversed(returns)):
        v[s] += alpha * (g - v[s])
    return v


def make(lam, alpha=0.3, **kw):
    return LinearTD(n_features=4, alpha=alpha, gamma=GAMMA, lam=lam, update_rule="true_online", **kw)


def test_true_online_lambda_zero_equals_classic_online_td0():
    v0 = np.array([0.2, -0.1, 0.4, 0.3])
    learner = make(0.0)
    learner.weights[:] = v0
    learner.update(Episode(features=STATES, rewards=REWARDS, terminated=True))
    np.testing.assert_allclose(
        learner.weights, classic_online_td0(STATES, REWARDS, True, 0.3, GAMMA, 4, v0)
    )


def test_true_online_lambda_one_ends_episode_with_monte_carlo_weights():
    v0 = np.array([0.2, -0.1, 0.4, 0.3])
    learner = make(1.0)
    learner.weights[:] = v0
    learner.update(Episode(features=STATES, rewards=REWARDS, terminated=True))
    np.testing.assert_allclose(
        learner.weights, sequential_mc(STATES[:-1], REWARDS, 0.3, GAMMA, 4, v0)
    )


def test_true_online_lambda_one_matches_offline_rule_on_terminated_episode():
    ep = Episode(features=STATES, rewards=REWARDS, terminated=True)
    online, offline = make(1.0), LinearTD(4, alpha=0.3, gamma=GAMMA, lam=1.0, update_rule="offline")
    online.update(ep)
    offline.update(ep)
    np.testing.assert_allclose(online.weights, offline.weights)


def test_true_online_truncated_step_bootstraps_from_final_state():
    learner = make(0.0, alpha=1.0)
    learner.weights[1] = 10.0
    learner.update(Episode(features=[0, 1], rewards=[1.0], terminated=False))
    assert learner.value(0) == pytest.approx(1.0 + GAMMA * 10.0)


def test_true_online_terminated_step_does_not_bootstrap():
    learner = make(0.0, alpha=1.0)
    learner.weights[1] = 10.0
    learner.update(Episode(features=[0, 1], rewards=[1.0], terminated=True))
    assert learner.value(0) == pytest.approx(1.0)


def test_true_online_values_change_during_the_episode():
    learner = make(0.0, alpha=1.0)
    learner.begin_episode(0)
    learner.observe(1.0, 1, terminated=False, truncated=False)
    assert learner.value(0) == pytest.approx(1.0)  # updated before the episode ends


def test_true_online_sarsa_updates_only_taken_action_with_next_action_bootstrap():
    learner = make(0.0, alpha=1.0, n_actions=2)
    learner.weights[1, 1] = 4.0
    learner.update(Episode(features=[0, 1], rewards=[0.0], actions=[0, 1], terminated=False))
    np.testing.assert_allclose(learner.q_values(0), [GAMMA * 4.0, 0.0])


def test_unknown_update_rule_is_rejected():
    with pytest.raises(ValueError):
        LinearTD(4, alpha=0.1, gamma=GAMMA, lam=0.5, update_rule="sideways")
