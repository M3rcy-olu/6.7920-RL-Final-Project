"""Tests for the chain-of-rooms maze (decision D13) and its 15-world catalog."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from minigrid.core.world_object import Door, Key, Wall

import mini_envs  # noqa: F401  (registers ids)
from mini_envs.chain_rooms import ChainRoomsEnv, plan, reachable_states, transition_table
from mini_envs.chain_worlds import CHAIN_WORLDS


def make(**kw):
    base = dict(rooms=("c1", "c2", "c1", "c2"), room_size=5, doors=2, corridor_len=3)
    base.update(kw)
    env = ChainRoomsEnv(**base)
    env.reset(seed=0)
    return env


def play(env, actions):
    rewards, terminated = [], False
    for a in actions:
        _, r, terminated, truncated, _ = env.step(a)
        rewards.append(r)
        assert not truncated
    return rewards, terminated


# ---- catalog --------------------------------------------------------------
def test_catalog_has_fifteen_registered_worlds():
    assert len(CHAIN_WORLDS) == 15
    for env_id in CHAIN_WORLDS:
        assert env_id in gym.registry


@pytest.mark.parametrize("env_id", sorted(CHAIN_WORLDS))
def test_every_catalog_world_is_solvable_with_full_return(env_id):
    env = gym.make(env_id).unwrapped
    env.reset(seed=0)
    actions = plan(env)
    env.reset(seed=0)
    rewards, terminated = play(env, actions)
    assert terminated and sum(rewards) == pytest.approx(1.0)


# ---- structure ------------------------------------------------------------
def test_layout_is_fixed_across_reset_seeds():
    env = make()
    a = env.grid.encode().copy()
    env.reset(seed=123)
    assert np.array_equal(a, env.grid.encode())


def test_action_space_excludes_drop():
    env = make()
    assert env.action_space.n == 5  # left, right, forward, pickup, toggle


def test_each_room_has_one_locked_door_per_key_colour():
    env = make(doors=3)
    doors = env.room_doors(0)
    keys = env.room_keys(0)
    assert len(doors) == len(keys) == 3
    assert {env.grid.get(*p).color for p in doors} == {env.grid.get(*p).color for p in keys}
    assert all(isinstance(env.grid.get(*p), Door) and env.grid.get(*p).is_locked for p in doors)


# ---- rewards and feedback placement ----------------------------------------
def test_feedback_every_room_pays_equal_pieces_summing_to_one():
    env = make(feedback_every=1)
    rewards, terminated = play(env, plan(env))
    paid = [r for r in rewards if r]
    assert terminated and paid == pytest.approx([0.25] * 4)


def test_feedback_every_two_rooms_pays_two_halves():
    env = make(feedback_every=2)
    paid = [r for r in play(env, plan(env))[0] if r]
    assert paid == pytest.approx([0.5, 0.5])


def test_end_only_feedback_pays_once_at_the_end():
    env = make(feedback_every=None)
    rewards, terminated = play(env, plan(env))
    assert terminated and [r for r in rewards if r] == pytest.approx([1.0])
    assert rewards[-1] == pytest.approx(1.0)


def test_wrong_door_in_case1_room_still_reaches_end_but_loses_that_gap():
    env = make(feedback_every=1)
    wrong = 1 - env.correct_door[0]
    actions = plan(env, door_choices={0: wrong})
    env.reset(seed=0)
    rewards, terminated = play(env, actions)
    assert terminated and sum(rewards) == pytest.approx(0.75)


def test_wrong_door_with_end_only_feedback_returns_zero():
    env = make(feedback_every=None)
    wrong = 1 - env.correct_door[2]
    actions = plan(env, door_choices={2: wrong})
    env.reset(seed=0)
    rewards, terminated = play(env, actions)
    assert terminated and sum(rewards) == 0.0


def test_any_door_is_correct_in_case2_room():
    env = make(feedback_every=1)
    for door in (0, 1):
        env.reset(seed=0)
        actions = plan(env, door_choices={1: door})
        env.reset(seed=0)
        assert sum(play(env, actions)[0]) == pytest.approx(1.0)


# ---- key and door mechanics --------------------------------------------------
def _open_first_door(env):
    """Play the optimal plan until room 0's door opens."""
    for a in plan(env):
        env.step(a)
        if env.opened is not None:
            return
    raise AssertionError("no door opened")


def test_used_key_disappears_other_doors_wall_off_and_leftover_keys_vanish():
    env = make(doors=3)
    doors, keys = env.room_doors(0), env.room_keys(0)
    _open_first_door(env)
    assert env.carrying is None
    opened = doors[env.opened]
    assert env.grid.get(*opened) is None  # doorway
    assert all(isinstance(env.grid.get(*p), Wall) for i, p in enumerate(doors) if i != env.opened)
    assert all(not isinstance(env.grid.get(*p), Key) for p in keys)


def test_entering_next_room_seals_the_way_back():
    env = make()
    actions = plan(env)
    for a in actions:
        env.step(a)
        if env.frontier == 1:
            break
    assert all(isinstance(env.grid.get(*p), Wall) for p in env.room_entries(1))


# ---- Markov state ------------------------------------------------------------
def test_markov_state_round_trip_reproduces_transitions():
    env = make()
    actions = plan(env)
    env.reset(seed=0)
    for a in actions[:15]:
        env.step(a)
    s = env.markov_state()
    expected = [env.step(a)[1:3] for a in actions[15:25]] + [env.markov_state()]
    env.set_markov_state(s)
    got = [env.step(a)[1:3] for a in actions[15:25]] + [env.markov_state()]
    assert got == expected


def test_restoring_a_carried_key_removes_it_from_the_floor():
    env = make()
    for a in plan(env):
        env.step(a)
        if env.carrying is not None:
            break
    live = env.grid.encode().copy()
    env.set_markov_state(env.markov_state())
    assert np.array_equal(env.grid.encode(), live)


def test_reachable_state_space_is_small_enough_for_tables():
    env = make()
    n = len(reachable_states(env))
    assert 100 < n < 20_000


# ---- compiled transition table -------------------------------------------------
@pytest.mark.parametrize("feedback_every", [1, 2, None])
def test_transition_table_matches_env_on_random_walks(feedback_every):
    env = make(feedback_every=feedback_every)
    table = transition_table(env)
    rng = np.random.default_rng(0)
    for ep in range(20):
        env.reset(seed=ep)
        s = table.start
        assert table.states[s] == env.markov_state()
        for _ in range(300):
            a = int(rng.integers(env.action_space.n))
            _, r, terminated, _, _ = env.step(a)
            assert table.reward[s, a] == pytest.approx(r)
            assert table.terminal[s, a] == terminated
            s = table.next[s, a]
            assert table.states[s] == env.markov_state()
            if terminated:
                break


def test_transition_table_follows_the_optimal_plan_to_full_return():
    env = make(feedback_every=1)
    table = transition_table(env)
    s, total = table.start, 0.0
    for a in plan(env):
        total += table.reward[s, a]
        done = table.terminal[s, a]
        s = table.next[s, a]
    assert done and total == pytest.approx(1.0)
