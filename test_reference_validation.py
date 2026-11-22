import numpy as np
import pytest
from environments import make_env
from reference_validation import exact_reference, compare, mc_update


@pytest.mark.parametrize('name', ['FrozenLake-v1', 'CliffWalking-v1'])
def test_exact_and_pairing(name):
    env = make_env(name)
    v, mask, residual = exact_reference(env.unwrapped)
    assert residual < 1e-9
    assert np.isfinite(v).all()
    assert len(mask) == (11 if name == 'FrozenLake-v1' else 37)
    if name == 'CliffWalking-v1':
        # Three wall/start actions cost -1, stepping right costs -100.
        expected_start = (-103 + .99*(v[24] + 3*v[36]))/4
        assert abs(v[36] - expected_start) < 1e-9
    rows = compare(name, 3, v, mask, budgets=(100, 200))
    assert rows == compare(name, 3, v, mask, budgets=(100, 200))
    for budget in (100, 200):
        group = [r for r in rows if r['transitions'] == budget]
        assert len({r['completed_episodes'] for r in group}) == 1
        assert len({r['visited_states'] for r in group}) == 1
    env.close()


def test_mc_known_returns():
    v, counts = np.zeros(3), np.zeros(3, dtype=int)
    mc_update(v, counts, [(0, 1), (1, 2)], gamma=.5)
    np.testing.assert_allclose(v, [2, 2, 0])
    mc_update(v, counts, [(0, 4)], gamma=.5)
    assert v[0] == 3


def test_analytic_one_state_reference():
    class Model:
        class observation_space:
            n = 1
        class action_space:
            n = 1
        initial_state_distrib = np.array([1.])
        P = {0: {0: [(1., 0, 1., False)]}}
    v, mask, residual = exact_reference(Model(), gamma=.9)
    np.testing.assert_allclose(v, [10.])
    assert mask == [0] and residual < 1e-12
