import numpy as np
import pytest
from environments import make_env
from rl_baseline import evaluate
from td_lambda import td_update


@pytest.mark.parametrize('name', ['FrozenLake-v1', 'CliffWalking-v1'])
def test_seeded_reproducibility(name):
    a = evaluate(name, episodes=3, seed=42)
    assert a == evaluate(name, episodes=3, seed=42)
    assert len(a['episodes']) == 3
    assert all(0 < ep['steps'] <= 100 for ep in a['episodes'])
    assert all(ep['terminated'] or ep['truncated'] for ep in a['episodes'])
    assert np.isfinite(a['values']).all()


def test_frozen_lake_known_safe_path():
    env = make_env('FrozenLake-v1', slippery=False)
    try:
        env.reset(seed=0)
        for action in [1, 1, 2, 1, 2, 2]:
            state, reward, term, trunc, _ = env.step(action)
        assert (state, reward, term, trunc) == (15, 1, True, False)
    finally:
        env.close()


def test_cliff_penalty_and_safe_path():
    env = make_env('CliffWalking-v1')
    try:
        env.reset(seed=0)
        state, reward, term, trunc, _ = env.step(1)
        assert (state, reward, term, trunc) == (36, -100, False, False)
        env.reset(seed=0)
        total = 0
        for action in [0] + [1] * 11 + [2]:
            state, reward, term, trunc, _ = env.step(action)
            total += reward
        assert (state, total, term, trunc) == (47, -13, True, False)
    finally:
        env.close()


def test_time_limit_is_truncation():
    env = make_env('FrozenLake-v1', slippery=False, max_steps=1)
    try:
        env.reset(seed=0)
        _, _, term, trunc, _ = env.step(0)
        assert trunc and not term
    finally:
        env.close()


def test_td_zero_is_one_step():
    v, z = np.array([0., 2.]), np.zeros(2)
    td_update(v, z, 0, 1, 1, False, alpha=.5, gamma=.9, lam=0)
    np.testing.assert_allclose(v, [1.4, 2.])
    np.testing.assert_array_equal(z, [0, 0])


def test_terminal_does_not_bootstrap():
    v, z = np.array([0., 100.]), np.zeros(2)
    td_update(v, z, 0, 1, 1, True, alpha=.5, gamma=.9, lam=.8)
    assert v[0] == .5


def test_trace_propagates_reward():
    v, z = np.zeros(3), np.zeros(3)
    td_update(v, z, 0, 0, 1, False, alpha=1, gamma=1, lam=.5)
    td_update(v, z, 1, 1, 2, True, alpha=1, gamma=1, lam=.5)
    np.testing.assert_allclose(v, [.5, 1, 0])
