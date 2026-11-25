"""Full-state encoding and policy integration for randomized GoToObject."""

import json

import pytest

from q_learning import Policy, evaluate, train
from q_learning.adapters import make_env, state

ENV_ID = "MiniGrid-Proj-GoToObj-v0"


def test_state_is_reproducible_and_captures_layout():
    env = make_env(ENV_ID, 256)
    try:
        obs, _ = env.reset(seed=0)
        original = state(obs, env)
        assert len(original) == 10 + 3 * 8 * 8
        assert all(type(value) is int for value in original)
        obs, _ = env.reset(seed=0)
        assert state(obs, env) == original
        obs, _ = env.reset(seed=1)
        assert state(obs, env) != original
    finally:
        env.close()


def test_state_distinguishes_target_pose_time_and_object_changes():
    env = make_env(ENV_ID, 256)
    try:
        obs, _ = env.reset(seed=0)
        base = env.unwrapped
        original = state(obs, env)
        base.agent_dir = (base.agent_dir + 1) % 4
        assert state(obs, env) != original
        obs, _ = env.reset(seed=0)
        base.step_count += 1
        assert state(obs, env) != original
        obs, _ = env.reset(seed=0)
        objects = [
            (x, y, base.grid.get(x, y))
            for x in range(1, base.width - 1)
            for y in range(1, base.height - 1)
            if base.grid.get(x, y) is not None
        ]
        other = next(o for o in objects if o[:2] != base.target_pos)
        base.target_pos = other[:2]
        base.target_color, base.target_type = other[2].color, other[2].type
        assert state(obs, env) != original
        obs, _ = env.reset(seed=0)
        x, y, obj = other
        base.grid.get(x, y).color = "red" if obj.color != "red" else "blue"
        assert state(obs, env) != original
        obs, _ = env.reset(seed=0)
        base.grid.set(x, y, None)
        assert state(obs, env) != original
    finally:
        env.close()


def test_policy_round_trip_and_frozen_rollout(tmp_path):
    env = make_env(ENV_ID, 256)
    try:
        obs, _ = env.reset(seed=0)
        policy = Policy(ENV_ID, 256, {state(obs, env): 0})
        path = tmp_path / "goto.json"
        policy.save(path)
        loaded = Policy.load(path)
        assert loaded == policy
        assert loaded.act(obs, env) == 0
        before = dict(loaded.actions)
        result = evaluate(loaded, episodes=1, seed=0)
        assert result["mean_steps"] == 1
        assert result["unseen_episodes"] == 1
        assert loaded.actions == before
    finally:
        env.close()


def test_reject_nonmovement_actions():
    import gymnasium as gym

    env = gym.make(ENV_ID, movement_only=False)
    try:
        obs, _ = env.reset(seed=0)
        with pytest.raises(ValueError, match="movement-only"):
            state(obs, env)
    finally:
        env.close()


@pytest.mark.parametrize("change", ["width", "action", "dimension"])
def test_malformed_goto_policy_rejected(tmp_path, change):
    env = make_env(ENV_ID, 256)
    try:
        obs, _ = env.reset(seed=0)
        encoded = list(state(obs, env))
        action = 0
        if change == "width":
            encoded.pop()
        elif change == "action":
            action = 3
        else:
            encoded[0] = 0
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "env_id": ENV_ID,
                    "max_steps": 256,
                    "actions": [[encoded, action]],
                }
            )
        )
        with pytest.raises(ValueError):
            Policy.load(path)
    finally:
        env.close()


def test_training_smoke_only():
    policy = train(ENV_ID, episodes=2, max_steps=5)
    assert policy.actions
    assert all(0 <= action < 3 for action in policy.actions.values())
