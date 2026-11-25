"""Correctness / smoke tests for the custom Minigrid and Miniworld envs."""
from __future__ import annotations

import numpy as np
import pytest

from mini_envs import CUSTOM_ENVS, OBS_MODES, library_of, list_env_ids, make_env
from mini_envs.wrappers import MiniWorldMissionWrapper

# Minigrid action ids
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE = range(7)
DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return np.array_equal(a, b)


def test_custom_envs_registered():
    assert set(CUSTOM_ENVS) == set(list_env_ids(custom_only=True))
    assert "MiniGrid-Empty-5x5-v0" in list_env_ids("minigrid")
    assert "MiniWorld-OneRoom-v0" in list_env_ids("miniworld")


@pytest.mark.parametrize("env_id", sorted(CUSTOM_ENVS))
def test_seeded_reset_and_random_rollout(env_id):
    env_a, env_b = make_env(env_id), make_env(env_id)
    try:
        obs_a, _ = env_a.reset(seed=123)
        obs_b, _ = env_b.reset(seed=123)
        assert _same(obs_a, obs_b)
        assert env_a.unwrapped.mission == env_b.unwrapped.mission
        assert env_a.observation_space.contains(obs_a)

        rng = np.random.default_rng(0)
        for _ in range(25):
            action = int(rng.integers(env_a.action_space.n))
            obs, reward, terminated, truncated, _ = env_a.step(action)
            assert env_a.observation_space.contains(obs)
            assert np.isfinite(reward)
            if terminated or truncated:
                env_a.reset(seed=124)
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.parametrize(
    "env_id,obs_mode",
    [(e, m) for e in sorted(CUSTOM_ENVS) for m in OBS_MODES[library_of(e)]],
)
def test_every_obs_mode_matches_its_space(env_id, obs_mode):
    env = make_env(env_id, obs=obs_mode)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_invalid_obs_mode_rejected():
    with pytest.raises(ValueError):
        make_env("MiniWorld-Proj-GoToObj-v0", obs="full")
    with pytest.raises(ValueError):
        make_env("CartPole-v1")


def test_keydoor_scripted_solution():
    """Fixed 8x8 layout: start (1,1) facing right, key (2,5), door (4,5), goal (6,6)."""
    env = make_env("MiniGrid-Proj-KeyDoor-v0")
    try:
        env.reset(seed=0)
        grid = env.unwrapped.grid
        assert grid.get(4, 5).type == "door" and grid.get(4, 5).is_locked
        assert grid.get(2, 5).type == "key"
        assert grid.get(6, 6).type == "goal"

        plan = [FORWARD, RIGHT, FORWARD, FORWARD, FORWARD, PICKUP,  # grab key
                LEFT, FORWARD, RIGHT, FORWARD, LEFT, TOGGLE,          # unlock door
                FORWARD, FORWARD, RIGHT, FORWARD, LEFT, FORWARD]      # walk to goal
        for i, action in enumerate(plan):
            _, reward, terminated, truncated, _ = env.step(action)
            assert not truncated
            assert terminated == (i == len(plan) - 1), f"step {i}"
        assert reward > 0
    finally:
        env.close()


def test_keydoor_randomized_layout_varies():
    env = make_env("MiniGrid-Proj-KeyDoor-Random-v0")
    try:
        layouts = set()
        for seed in range(10):
            env.reset(seed=seed)
            layouts.add((tuple(env.unwrapped.agent_pos), env.unwrapped.grid.encode().tobytes()))
        assert len(layouts) > 1
    finally:
        env.close()


def _face_from_neighbor(env, target):
    """Put the agent next to ``target`` so that one RIGHT turn makes it face the target."""
    u = env.unwrapped
    for d, (dx, dy) in DIR_VEC.items():
        cell = (target[0] - dx, target[1] - dy)  # agent here, facing dir d, sees target
        if u.grid.get(*cell) is None:
            u.agent_pos = cell
            u.agent_dir = (d - 1) % 4
            return
    raise AssertionError("target has no free neighbor")


def test_grid_goto_success_and_wrong_object():
    env = make_env("MiniGrid-Proj-GoToObj-v0", num_objs=3)
    try:
        env.reset(seed=3)
        u = env.unwrapped
        assert u.mission == f"go to the {u.target_color} {u.target_type}"

        # Facing a non-target object does not end the episode.
        others = [
            (x, y)
            for x in range(u.width)
            for y in range(u.height)
            if (obj := u.grid.get(x, y)) is not None
            and obj.type in ("ball", "box", "key")
            and (x, y) != u.target_pos
        ]
        _face_from_neighbor(env, others[0])
        _, reward, terminated, _, _ = env.step(RIGHT)
        assert not terminated and reward == 0

        _face_from_neighbor(env, u.target_pos)
        _, reward, terminated, _, _ = env.step(RIGHT)
        assert terminated and reward > 0
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["MiniWorld-Proj-GoToObj-v0", "MiniWorld-Proj-TwoRoom-v0"])
def test_world_reaching_target_terminates_with_reward(env_id):
    env = make_env(env_id)
    try:
        env.reset(seed=0)
        u = env.unwrapped
        target = u.target if env_id.endswith("GoToObj-v0") else u.box
        _, reward, terminated, _, _ = env.step(0)
        assert not terminated and reward == 0  # starts away from target

        u.agent.pos = target.pos + np.array([target.radius + u.agent.radius + 0.05, 0, 0])
        _, reward, terminated, _, _ = env.step(0)  # turning still triggers the near() check
        assert terminated and reward > 0
    finally:
        env.close()


def test_world_goto_missions_vary_and_match_info():
    env = make_env("MiniWorld-Proj-GoToObj-v0", obs="dict")
    try:
        missions = set()
        for seed in range(12):
            obs, info = env.reset(seed=seed)
            assert obs["mission"] == info["mission"] == env.unwrapped.mission
            kinds = [(c, t) for c, t, _ in env.unwrapped.objects]
            assert len(set(kinds)) == len(kinds) == 3
            missions.add(obs["mission"])
        assert len(missions) > 3
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["MiniGrid-Proj-KeyDoor-v0", "MiniWorld-Proj-TwoRoom-v0"])
def test_max_steps_is_truncation(env_id):
    env = make_env(env_id, max_steps=1)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated and not terminated
    finally:
        env.close()


def test_mission_wrapper_on_builtin_env():
    env = MiniWorldMissionWrapper(make_env("MiniWorld-OneRoom-v0"))
    try:
        obs, _ = env.reset(seed=0)
        assert obs["mission"] == "" and obs["image"].shape == (60, 80, 3)
        assert env.observation_space.contains(obs)
    finally:
        env.close()
