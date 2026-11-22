"""Tests for the offline lambda-return update rule of LinearTD (predict and control modes)."""
from __future__ import annotations

import numpy as np
import pytest

from td.learner import Episode, LinearTD
from td.policies import epsilon_greedy

GAMMA = 0.9


def test_predict_lambda_one_alpha_one_sets_values_to_monte_carlo_returns():
    learner = LinearTD(update_rule="offline", n_features=4, alpha=1.0, gamma=GAMMA, lam=1.0)
    ep = Episode(features=[0, 1, 2, 3], rewards=[0.0, 0.0, 1.0], terminated=True)
    learner.update(ep)
    np.testing.assert_allclose(
        [learner.value(s) for s in (0, 1, 2)], [GAMMA**2, GAMMA, 1.0]
    )


def test_targets_use_values_from_start_of_episode():
    learner = LinearTD(update_rule="offline", n_features=3, alpha=1.0, gamma=GAMMA, lam=0.0)
    learner.weights[1] = 5.0
    ep = Episode(features=[0, 1, 2], rewards=[0.0, 2.0], terminated=True)
    learner.update(ep)
    assert learner.value(0) == pytest.approx(GAMMA * 5.0)  # old V(1), not the new 2.0
    assert learner.value(1) == pytest.approx(2.0)


def test_truncated_episode_bootstraps_from_final_state_value():
    learner = LinearTD(update_rule="offline", n_features=2, alpha=1.0, gamma=GAMMA, lam=1.0)
    learner.weights[1] = 10.0
    learner.update(Episode(features=[0, 1], rewards=[1.0], terminated=False))
    assert learner.value(0) == pytest.approx(1.0 + GAMMA * 10.0)


def test_terminated_episode_ignores_final_state_value():
    learner = LinearTD(update_rule="offline", n_features=2, alpha=1.0, gamma=GAMMA, lam=1.0)
    learner.weights[1] = 10.0
    learner.update(Episode(features=[0, 1], rewards=[1.0], terminated=True))
    assert learner.value(0) == pytest.approx(1.0)


def test_control_updates_only_the_taken_action_and_bootstraps_on_next_action():
    learner = LinearTD(update_rule="offline", n_features=2, n_actions=2, alpha=1.0, gamma=GAMMA, lam=0.0)
    learner.weights[1, 1] = 4.0  # Q(s=1, a=1)
    ep = Episode(features=[0, 1], rewards=[0.0], actions=[0, 1], terminated=False)
    learner.update(ep)
    assert learner.q_values(0)[0] == pytest.approx(GAMMA * 4.0)  # SARSA: Q(S1, A1)
    assert learner.q_values(0)[1] == 0.0


def test_step_size_is_split_across_active_features():
    learner = LinearTD(update_rule="offline", n_features=4, alpha=1.0, gamma=GAMMA, lam=0.0)
    learner.update(Episode(features=[np.array([0, 2]), 3], rewards=[1.0], terminated=True))
    np.testing.assert_allclose(learner.weights, [0.5, 0.0, 0.5, 0.0])
    assert learner.value(np.array([0, 2])) == pytest.approx(1.0)


def test_dense_features_use_gradient_step():
    learner = LinearTD(update_rule="offline", n_features=2, alpha=0.1, gamma=GAMMA, lam=0.0)
    learner.update(Episode(features=[np.array([1.0, 2.0]), np.zeros(2)], rewards=[1.0], terminated=True))
    np.testing.assert_allclose(learner.weights, [0.1, 0.2])


def test_episode_lengths_are_validated():
    with pytest.raises(ValueError):
        Episode(features=[0, 1], rewards=[0.0, 1.0], terminated=True)
    with pytest.raises(ValueError):
        Episode(features=[0, 1], rewards=[0.0], actions=[0], terminated=True)


def test_control_mode_requires_actions():
    learner = LinearTD(update_rule="offline", n_features=2, n_actions=2, alpha=1.0, gamma=GAMMA, lam=0.0)
    with pytest.raises(ValueError):
        learner.update(Episode(features=[0, 1], rewards=[0.0], terminated=True))


def test_epsilon_zero_is_greedy():
    rng = np.random.default_rng(0)
    assert all(epsilon_greedy(np.array([0.0, 3.0, 1.0]), 0.0, rng) == 1 for _ in range(20))


def test_greedy_ties_are_broken_randomly():
    rng = np.random.default_rng(0)
    picks = {epsilon_greedy(np.array([1.0, 1.0, 0.0]), 0.0, rng) for _ in range(50)}
    assert picks == {0, 1}


def test_epsilon_one_is_uniform_over_all_actions():
    rng = np.random.default_rng(0)
    picks = {epsilon_greedy(np.array([0.0, 9.0, 0.0]), 1.0, rng) for _ in range(100)}
    assert picks == {0, 1, 2}
