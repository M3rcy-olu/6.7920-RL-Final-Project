"""Q-learning baseline policies on the chain-of-rooms worlds (D2, D4, D13)."""
from __future__ import annotations

import gymnasium as gym
import pytest

import mini_envs  # noqa: F401  (registers ids)
from mini_envs.chain_rooms import plan, reachable_states
from q_learning import Policy, evaluate, train
from q_learning.adapters import ENVS, make_env, state
from q_learning.core import truncation_is_terminal

SMALL = "MiniGrid-Proj-Chain-Rooms-2-v0"


def test_every_chain_and_cost_world_is_a_supported_baseline():
    from mini_envs.chain_worlds import CHAIN_WORLDS, COST_WORLDS

    assert len(CHAIN_WORLDS) + len(COST_WORLDS) == 18
    assert set(CHAIN_WORLDS) | set(COST_WORLDS) <= set(ENVS)


def test_chain_state_is_six_ints_without_the_loss_field():
    env = make_env(SMALL, 100)
    try:
        obs, _ = env.reset(seed=0)
        s = state(obs, env)
        assert len(s) == 6 and all(type(v) is int for v in s)
        full = env.unwrapped.markov_state()
        assert s[:3] == full[:3] and s[4] == full[4]
    finally:
        env.close()


@pytest.mark.parametrize("env_id", [SMALL, "MiniGrid-Proj-Chain-Base-v0", "MiniGrid-Proj-Chain-Cost-Graded-v0"])
def test_dropping_loss_loses_nothing_with_feedback_after_every_room(env_id):
    # With feedback after every room the loss field is fixed by (frontier, opened
    # door), so the 6-field state is still Markov for training.
    env = gym.make(env_id).unwrapped
    states = reachable_states(env)
    reduced = {s[:-1] for s in states}
    assert len(reduced) == len(states)


def test_chain_truncation_bootstraps_but_keydoor_truncation_ends_the_task():
    assert not truncation_is_terminal(SMALL)
    assert not truncation_is_terminal("CliffWalking-v1")
    assert truncation_is_terminal("MiniGrid-Proj-KeyDoor-v0")


@pytest.fixture(scope="module")
def small_policy():
    return train(SMALL, episodes=300, max_steps=3600, seed=0)


def test_trained_policy_solves_the_small_world(small_policy):
    result = evaluate(small_policy, episodes=5)
    assert result["success_rate"] == 1.0
    assert result["mean_return"] == pytest.approx(1.0)
    assert result["unseen_episodes"] == 0


def test_policy_covers_every_reachable_state(small_policy):
    env = gym.make(SMALL).unwrapped
    covered = set(small_policy.actions)
    env_wrapped = make_env(SMALL, 100)
    try:
        for s in reachable_states(env):
            if s[4] == env.n_rooms:  # terminal states never need an action
                continue
            env.set_markov_state(s)
            assert state(None, _Bare(env)) in covered
    finally:
        env_wrapped.close()


def test_policy_round_trips_through_json(small_policy, tmp_path):
    path = tmp_path / "chain.json"
    small_policy.save(path)
    assert Policy.load(path) == small_policy


def test_policy_runs_under_other_feedback_settings(small_policy):
    # The frozen policy ignores the loss field, so it acts in end-only worlds too.
    env = gym.make(SMALL, feedback_every=None, max_steps=small_policy.max_steps)
    try:
        obs, _ = env.reset(seed=0)
        total, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, _ = env.step(small_policy.act(obs, env))
            total += r
            done = terminated or truncated
        assert terminated and total == pytest.approx(1.0)
    finally:
        env.close()


def test_greedy_policy_is_close_to_the_optimal_plan(small_policy):
    env = make_env(SMALL, 3600)
    try:
        obs, _ = env.reset(seed=0)
        steps = 0
        while True:
            obs, _, terminated, truncated, _ = env.step(small_policy.act(obs, env))
            steps += 1
            if terminated or truncated:
                break
        # 300 episodes give a near-optimal, not exactly optimal, path.
        assert terminated and steps <= 1.25 * len(plan(env.unwrapped))
    finally:
        env.close()


class _Bare:
    """Minimal stand-in so ``state`` can read an unwrapped env directly."""

    def __init__(self, unwrapped):
        self.unwrapped = unwrapped
        self.spec = gym.spec(SMALL)
