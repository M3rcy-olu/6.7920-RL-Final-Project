"""Tests for exact Q* and state importance on the chain worlds (test 3)."""
from __future__ import annotations

import numpy as np
import pytest

import mini_envs  # noqa: F401  (registers ids)
from mini_envs.chain_rooms import ChainRoomsEnv
from mini_envs.chain_worlds import CHAIN_WORLDS
from mini_envs.criticality import (
    BANDS,
    check_greedy_matches_plan,
    compute,
    percentile_rank,
    standard_env,
)

BASE = "MiniGrid-Proj-Chain-Base-v0"


@pytest.fixture(scope="module")
def base():
    env = standard_env(BASE)
    return env, compute(env)


def test_requires_end_only_reward():
    env = ChainRoomsEnv(feedback_every=1)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        compute(env)


@pytest.mark.parametrize("env_id", sorted(CHAIN_WORLDS))
def test_greedy_q_star_follows_shortest_plan(env_id):
    env = standard_env(env_id)
    check_greedy_matches_plan(env, compute(env))


def test_q_star_bellman_consistent(base):
    env, crit = base
    for i in np.random.default_rng(0).choice(len(crit.states), 50, replace=False):
        s = crit.states[i]
        for a in range(env.action_space.n):
            env.set_markov_state(s)
            env._skip_obs = True
            _, r, term, _, _ = env.step(a)
            env._skip_obs = False
            target = r if term else r + crit.gamma * crit.q[crit.index[env.markov_state()]].max()
            assert crit.q[i, a] == pytest.approx(target, abs=1e-9)
    env.reset(seed=0)


def test_most_critical_states_are_in_case1_rooms(base):
    env, crit = base
    live = np.where(~np.isnan(crit.rank))[0]
    top = live[np.argsort(-crit.si[live])][:6]
    assert all(env.room_types[crit.states[i][4]] == "c1" for i in top)


def test_no_choice_matters_world_has_no_large_si():
    env = standard_env("MiniGrid-Proj-Chain-Matter-0of4-v0")
    crit = compute(env)
    assert crit.si.max() < 1e-3


def test_lost_states_are_unranked(base):
    _, crit = base
    # Lost = success no longer possible (V* = 0). That covers the mistake flag,
    # and also holding the wrong key in a case-1 room before the door opens.
    lost = crit.q.max(axis=1) == 0
    mistake = np.array([s[6] for s in crit.states]) > 0  # loss so far in the gap
    assert np.all(lost[mistake])
    assert np.all(np.isnan(crit.rank[lost]))
    assert not np.any(np.isnan(crit.rank[~lost]))


def test_bands_are_disjoint_and_non_empty(base):
    _, crit = base
    sets = {name: crit.band(name) for name in BANDS}
    for name, s in sets.items():
        assert s, name
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not sets[a] & sets[b]


def test_percentile_rank_ties_share_rank():
    r = percentile_rank(np.array([0.0, 0.0, 1.0, 2.0]))
    assert r.tolist() == [0.5, 0.5, 0.75, 1.0]


# ---- partial-cost rooms ------------------------------------------------------
from mini_envs.chain_rooms import plan  # noqa: E402
from mini_envs.chain_worlds import COST_WORLDS  # noqa: E402


@pytest.mark.parametrize("cost", [0.25, 0.5])
def test_wrong_door_in_partial_cost_room_costs_that_fraction(cost):
    env = ChainRoomsEnv(rooms=(cost, "c2"), feedback_every=None, corridor_len=2)
    env.reset(seed=0)
    wrong = 1 - env.correct_door[0]
    total = sum(env.step(a)[1] for a in plan(env, {0: wrong}))
    assert total == pytest.approx(1 - cost)


def test_costs_add_within_a_gap_and_clip_at_zero():
    env = ChainRoomsEnv(rooms=(0.5, 0.75), feedback_every=None, corridor_len=2)
    env.reset(seed=0)
    wrong = {i: 1 - env.correct_door[i] for i in range(2)}
    assert sum(env.step(a)[1] for a in plan(env, wrong)) == 0.0


def test_c1_and_c2_rooms_unchanged_by_cost_rule():
    env = ChainRoomsEnv(rooms=("c1", "c2"), feedback_every=None, corridor_len=2)
    assert env.wrong_cost == [1.0, 0.0]
    assert env.room_types == ("c1", "c2")


@pytest.mark.parametrize("env_id", sorted(COST_WORLDS))
def test_cost_world_si_follows_room_cost(env_id):
    env = standard_env(env_id)
    crit = compute(env)
    check_greedy_matches_plan(env, crit)
    live = ~np.isnan(crit.rank)
    peak = [crit.si[live & (np.array([s[4] for s in crit.states]) == i)].max() for i in range(env.n_rooms)]
    order = np.argsort(env.wrong_cost)
    assert np.all(np.diff(np.array(peak)[order]) > 0)


def test_log_band_high_holds_only_costly_decisions():
    env = standard_env("MiniGrid-Proj-Chain-Cost-Graded-v0")
    crit = compute(env)
    rooms = {s[4] for s in crit.log_band("high")}
    assert rooms and all(env.wrong_cost[r] >= 0.5 for r in rooms)
