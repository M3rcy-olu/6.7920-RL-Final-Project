"""Learning, serialization and frozen rollout regression tests."""

import json
from pathlib import Path

import pytest

from q_learning import Policy, evaluate, train
from q_learning.adapters import make_env, state

POLICIES = Path(__file__).resolve().parents[1] / "q_learning/policies"


def test_round_trip(tmp_path):
    policy = Policy("FrozenLake-v1", 50, {(0,): 2, (1,): 1})
    path = tmp_path / "policy.json"
    policy.save(path)
    loaded = Policy.load(path)
    assert loaded == policy
    assert json.loads(path.read_text())["version"] == 1


def test_unknown_format_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"version": 99}')
    with pytest.raises(ValueError, match="version"):
        Policy.load(path)


def test_unknown_state_is_not_silent_random_fallback():
    env = make_env("FrozenLake-v1", 20)
    try:
        obs, _ = env.reset(seed=0)
        with pytest.raises(KeyError):
            Policy("FrozenLake-v1", 20, {}).act(obs, env)
    finally:
        env.close()


def test_training_reproducible():
    kwargs = {"episodes": 50, "max_steps": 30, "seed": 7}
    assert train("CliffWalking-v1", **kwargs) == train("CliffWalking-v1", **kwargs)


def test_keydoor_state_captures_time_direction_and_objects():
    env = make_env("MiniGrid-Proj-KeyDoor-v0", 100)
    try:
        obs, _ = env.reset(seed=0)
        original = state(obs, env)
        base = env.unwrapped
        base.agent_dir = 1
        assert state(obs, env) != original
        base.agent_dir = 0
        base.step_count += 1
        assert state(obs, env) != original
        base.step_count = 0
        key = base.grid.get(2, 5)
        base.grid.set(2, 5, None)
        base.carrying = key
        assert state(obs, env) != original
        base.carrying = None
        base.grid.set(2, 5, key)
        base.grid.get(4, 5).is_locked = False
        assert state(obs, env) != original
    finally:
        env.close()


@pytest.mark.parametrize("name", ["cliff_walking", "frozen_lake", "key_door"])
def test_saved_policy_is_frozen_and_usable(name):
    policy = Policy.load(POLICIES / f"{name}.json")
    before = dict(policy.actions)
    result = evaluate(policy, episodes=100)
    assert result["unseen_episodes"] == 0
    assert result["success_rate"] >= (0.6 if name == "frozen_lake" else 1.0)
    assert policy.actions == before


def test_q_learning_prefers_delayed_return(monkeypatch):
    import gymnasium as gym

    from q_learning import core

    class Fork(gym.Env):
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            self.position = 0
            return 0, {}

        def step(self, action):
            if self.position == 0:
                self.position = 1
                return 1, float(action == 0), action == 0, False, {}
            return 2, 3.0, True, False, {}

    monkeypatch.setattr(core, "make_env", lambda *args: Fork())
    monkeypatch.setattr(core, "state", lambda obs, env: (obs,))
    policy = train("test", episodes=500, seed=0)
    assert policy.actions[(0,)] == 1
    assert policy.actions[(1,)] in (0, 1)


def test_environment_and_horizon_must_match():
    env = make_env("MiniGrid-Proj-KeyDoor-v0", 100)
    try:
        obs, _ = env.reset(seed=0)
        with pytest.raises(ValueError, match="environment"):
            Policy("FrozenLake-v1", 100, {}).act(obs, env)
        with pytest.raises(ValueError, match="horizon"):
            Policy("MiniGrid-Proj-KeyDoor-v0", 256, {}).act(obs, env)
    finally:
        env.close()


@pytest.mark.parametrize("actions", [[[[0], 4]], [[[0], 1], [[0], 2]], [[[0.5], 1]]])
def test_malformed_action_table_rejected(tmp_path, actions):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "env_id": "FrozenLake-v1",
                "max_steps": 100,
                "actions": actions,
            }
        )
    )
    with pytest.raises(ValueError):
        Policy.load(path)


def test_evaluation_reports_unseen_without_taking_an_action():
    result = evaluate(Policy("FrozenLake-v1", 20, {}), episodes=3)
    assert result["unseen_episodes"] == 3
    assert result["success_rate"] == 0
    assert result["mean_steps"] == 0


def test_standard_online_update_without_replay(monkeypatch):
    import gymnasium as gym

    from q_learning import core

    class OneStep(gym.Env):
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return 0, {}

        def step(self, action):
            return 1, 1.0, True, False, {}

    seen = []

    def choose(values, epsilon, rng):
        seen.append(values.copy())
        return 0

    monkeypatch.setattr(core, "make_env", lambda *args: OneStep())
    monkeypatch.setattr(core, "state", lambda obs, env: (obs,))
    monkeypatch.setattr(core, "epsilon_greedy", choose)
    train("test", episodes=2, alpha=0.2)
    assert len(seen) == 2
    assert seen[0].tolist() == [0.0, 0.0]
    assert seen[1].tolist() == pytest.approx([0.2, 0.0])


def test_fixed_keydoor_baseline_is_solvable():
    # Diagnostic environment test only. Never supplied to Q-learning training.
    env = make_env("MiniGrid-Proj-KeyDoor-v0", 256)
    try:
        env.reset(seed=0)
        for action in [2, 1, 2, 2, 2, 3, 2, 0, 2, 5, 2, 2, 2, 1, 2]:
            _, reward, terminated, truncated, _ = env.step(action)
        assert terminated and not truncated
        assert reward > 0
    finally:
        env.close()


@pytest.mark.parametrize("env_id, expected", [("test", 0.5), ("MiniGrid-test", 0.0)])
def test_truncation_bootstrap_depends_on_task(monkeypatch, env_id, expected):
    import gymnasium as gym

    from q_learning import core

    class Cutoff(gym.Env):
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return 0, {}

        def step(self, action):
            return 0, 0.0, False, True, {}

    seen = []

    def choose(values, epsilon, rng):
        seen.append(values.copy())
        return 0

    monkeypatch.setattr(core, "make_env", lambda *args: Cutoff())
    monkeypatch.setattr(core, "state", lambda obs, env: (obs,))
    monkeypatch.setattr(core, "epsilon_greedy", choose)
    train(env_id, episodes=2, alpha=1, gamma=0.5, initial_value=1)
    assert seen[0].tolist() == [1.0, 1.0]
    assert seen[1].tolist() == [expected, 1.0]
