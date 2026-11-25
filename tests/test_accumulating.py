"""Tests for the default update rule: online TD(lambda) / Sarsa(lambda) with accumulating traces (D3)."""
from __future__ import annotations

import numpy as np
import pytest

from td.learner import Episode, LinearTD

GAMMA = 0.9
STATES = [0, 1, 0, 2, 1, 3]  # revisits states 0 and 1
REWARDS = [0.0, 1.0, 0.0, 0.5, 2.0]


def classic_online_td0(states, rewards, alpha, gamma, n, v0):
    v = v0.copy()
    for t, r in enumerate(rewards):
        last = t == len(rewards) - 1
        v_next = 0.0 if last else v[states[t + 1]]
        v[states[t]] += alpha * (r + gamma * v_next - v[states[t]])
    return v


def make(lam, alpha=0.3, **kw):
    return LinearTD(n_features=4, alpha=alpha, gamma=GAMMA, lam=lam, **kw)


def test_accumulating_is_the_default_update_rule():
    assert make(0.5).update_rule == "accumulating"


def test_lambda_zero_equals_classic_online_td0():
    v0 = np.array([0.2, -0.1, 0.4, 0.3])
    learner = make(0.0)
    learner.weights[:] = v0
    learner.update(Episode(features=STATES, rewards=REWARDS, terminated=True))
    np.testing.assert_allclose(learner.weights, classic_online_td0(STATES, REWARDS, 0.3, GAMMA, 4, v0))


def test_trace_propagates_reward_to_earlier_state():
    # Same case as test_trace_propagates_reward on jenny/rl-bootstrap.
    learner = LinearTD(n_features=3, alpha=1.0, gamma=1.0, lam=0.5)
    learner.update(Episode(features=[0, 1, 2], rewards=[0.0, 1.0], terminated=True))
    np.testing.assert_allclose(learner.weights, [0.5, 1.0, 0.0])


def test_lambda_one_equals_monte_carlo_when_no_state_repeats():
    learner = LinearTD(n_features=5, alpha=0.3, gamma=GAMMA, lam=1.0)
    learner.update(Episode(features=[0, 1, 2, 3, 4], rewards=[0.0, 1.0, 0.0, 2.0], terminated=True))
    returns = [GAMMA * 1 + GAMMA**3 * 2, 1 + GAMMA**2 * 2, GAMMA * 2, 2.0]
    np.testing.assert_allclose(learner.weights[:4], [0.3 * g for g in returns])


def test_lambda_one_differs_from_monte_carlo_when_states_repeat():
    # Documents D3: TD(1) with accumulating traces only approximates MC under revisits.
    online, offline = make(1.0), make(1.0, update_rule="offline")
    ep = Episode(features=STATES, rewards=REWARDS, terminated=True)
    online.update(ep)
    offline.update(ep)
    assert not np.allclose(online.weights, offline.weights)


def test_truncated_episode_bootstraps_from_final_state():
    learner = make(0.0, alpha=1.0)
    learner.weights[1] = 10.0
    learner.update(Episode(features=[0, 1], rewards=[1.0], terminated=False))
    assert learner.value(0) == pytest.approx(1.0 + GAMMA * 10.0)


def test_terminated_episode_does_not_bootstrap():
    learner = make(0.0, alpha=1.0)
    learner.weights[1] = 10.0
    learner.update(Episode(features=[0, 1], rewards=[1.0], terminated=True))
    assert learner.value(0) == pytest.approx(1.0)


def test_sarsa_updates_only_taken_action_with_next_action_bootstrap():
    learner = make(0.0, alpha=1.0, n_actions=2)
    learner.weights[1, 1] = 4.0
    learner.update(Episode(features=[0, 1], rewards=[0.0], actions=[0, 1], terminated=False))
    np.testing.assert_allclose(learner.q_values(0), [GAMMA * 4.0, 0.0])


def test_traces_reset_between_episodes():
    learner = LinearTD(n_features=3, alpha=1.0, gamma=1.0, lam=1.0)
    learner.update(Episode(features=[0, 1], rewards=[0.0], terminated=True))
    learner.update(Episode(features=[2, 1], rewards=[1.0], terminated=True))
    assert learner.value(0) == 0.0  # state 0's trace did not carry into episode 2
