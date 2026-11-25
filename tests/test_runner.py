"""End-to-end tests for the TD(lambda) runner on environments with adapters."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from td.adapters import make_adapter
from td.runner import run


def random_policy_values(env_id: str, gamma: float, **make_kwargs) -> np.ndarray:
    """Exact V^pi for the uniform-random policy, by iterative policy evaluation on env.P."""
    env = gym.make(env_id, **make_kwargs).unwrapped
    n_s, n_a = env.observation_space.n, env.action_space.n
    v = np.zeros(n_s)
    for _ in range(10_000):
        new = np.zeros(n_s)
        for s in range(n_s):
            for a in range(n_a):
                for p, s2, r, done in env.P[s][a]:
                    new[s] += p / n_a * (r + (0.0 if done else gamma * v[s2]))
        if np.max(np.abs(new - v)) < 1e-12:
            break
        v = new
    env.close()
    return v


@pytest.mark.parametrize("update_rule", ["accumulating", "true_online", "offline"])
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_predict_mode_recovers_exact_random_policy_values_on_frozenlake(lam, update_rule):
    gamma = 0.9
    truth = random_policy_values("FrozenLake-v1", gamma)
    # The goal is rarely reached by a random policy, so values converge slowly:
    # at 4k episodes RMSE is ~0.03-0.04, at 20k it is 0.004-0.018 over seeds 0-2.
    # 0.025 is ~15% of the typical nonterminal true value, so a real bias fails.
    result = run("FrozenLake-v1", mode="predict", lam=lam, alpha=0.005, gamma=gamma,
                 episodes=20_000, seed=0, max_steps=200, update_rule=update_rule)
    assert result["update_rule"] == update_rule
    nonterminal = truth > 0  # holes and the goal stay at 0
    rmse = np.sqrt(np.mean((np.array(result["values"])[nonterminal] - truth[nonterminal]) ** 2))
    assert rmse < 0.025, rmse


def test_control_mode_improves_return_on_cliffwalking():
    result = run("CliffWalking-v1", mode="control", lam=0.5, alpha=0.5, gamma=0.99,
                 epsilon=0.1, episodes=300, seed=0, max_steps=100)
    returns = [ep["return_"] for ep in result["episodes"]]
    assert np.mean(returns[-50:]) > np.mean(returns[:50]) + 20


def test_same_seed_gives_identical_runs():
    kwargs = dict(mode="control", lam=0.8, alpha=0.1, gamma=0.99, epsilon=0.2,
                  episodes=20, seed=7, max_steps=100)
    assert run("FrozenLake-v1", **kwargs) == run("FrozenLake-v1", **kwargs)


def test_truncation_fraction_is_reported():
    result = run("CliffWalking-v1", mode="predict", lam=0.0, alpha=0.1, gamma=0.99,
                 episodes=10, seed=0, max_steps=5)
    truncated = sum(ep["truncated"] for ep in result["episodes"])
    assert result["truncation_fraction"] == truncated / 10
    assert result["truncation_fraction"] > 0


def test_environments_without_an_adapter_are_rejected_clearly():
    with pytest.raises(NotImplementedError, match="adapter"):
        make_adapter("MiniWorld-Proj-TwoRoom-v0", max_steps=10)
