"""Off-policy tabular learning, with separate read-only evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from td.policies import epsilon_greedy

from .adapters import CHAIN_ENVS, ENVS, KEY_DOOR, live_states, make_env, spec, state


def truncation_is_terminal(env_id):
    """True only where the step limit is part of the task (KeyDoor keeps time in
    its state and its reward depends on it). Elsewhere a cutoff bootstraps (D4)."""
    return env_id == KEY_DOOR


@dataclass
class Policy:
    env_id: str
    max_steps: int
    actions: dict[tuple, int]

    def act(self, obs, env):
        """Fail explicitly on incompatible environments and unseen states."""
        if env.spec.id != self.env_id:
            raise ValueError("Policy and environment do not match")
        if (
            truncation_is_terminal(self.env_id)
            and env.unwrapped.max_steps != self.max_steps
        ):
            raise ValueError("KeyDoor policy requires its training horizon")
        action = self.actions[state(obs, env)]
        if not env.action_space.contains(action):
            raise ValueError("Policy contains an invalid action")
        return action

    def save(self, path):
        data = {
            "version": 1,
            "env_id": self.env_id,
            "max_steps": self.max_steps,
            "actions": [[list(s), a] for s, a in sorted(self.actions.items())],
        }
        Path(path).write_text(json.dumps(data, separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["version"] != 1:
            raise ValueError("Unsupported policy version")
        env_id, max_steps = data["env_id"], data["max_steps"]
        if env_id not in ENVS or type(max_steps) is not int or max_steps < 1:
            raise ValueError("Invalid policy environment or horizon")
        width, n_actions = spec(env_id)
        actions = {}
        for s, action in data["actions"]:
            if (
                len(s) != width
                or any(type(v) is not int for v in s)
                or type(action) is not int
                or not 0 <= action < n_actions
            ):
                raise ValueError("Invalid state or action in policy")
            key = tuple(s)
            if key in actions:
                raise ValueError("Duplicate state in policy")
            actions[key] = action
        return cls(env_id, max_steps, actions)


def train(
    env_id,
    *,
    episodes=20000,
    max_steps=256,
    seed=0,
    alpha=0.2,
    gamma=0.99,
    initial_value=0.0,
    progress=False,
):
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError("Invalid alpha or gamma")
    if not np.isfinite(initial_value):
        raise ValueError("initial_value must be finite")
    rng = np.random.default_rng(seed)
    env = make_env(env_id, max_steps)
    q = {}
    successes = 0

    def values(s):
        if s not in q:
            q[s] = np.full(env.action_space.n, initial_value, dtype=float)
        return q[s]

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            s = state(obs, env)
            epsilon = max(0.05, 1 - episode / (episodes * 0.8))
            while True:
                action = epsilon_greedy(values(s), epsilon, rng)
                obs, reward, terminated, truncated, _ = env.step(action)
                next_s = state(obs, env)
                # KeyDoor is a finite-horizon task with time in its state.
                finite_end = truncated and truncation_is_terminal(env_id)
                target = reward
                if not terminated and not finite_end:
                    target += gamma * values(next_s).max()
                values(s)[action] += alpha * (target - values(s)[action])
                s = next_s
                if terminated or truncated:
                    successes += int(terminated and reward > 0)
                    break
            if progress and (episode + 1) % 1000 == 0:
                print(
                    f"Episodes {episode + 1}: goal rewards={successes}, states={len(q)}",
                    flush=True,
                )
        # Chain worlds: give unvisited states the tie-break action, so the frozen
        # epsilon-greedy policy and exact DP are defined on every state.
        for s in live_states(env_id, env):
            values(s)
        return Policy(env_id, max_steps, {s: int(np.argmax(v)) for s, v in q.items()})
    finally:
        env.close()


def evaluate(policy, *, episodes=100, seed=100000):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = make_env(policy.env_id, policy.max_steps)
    returns, lengths = [], []
    successes = unseen = 0
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            steps = 0
            for _ in range(policy.max_steps):
                try:
                    action = policy.act(obs, env)
                except KeyError:
                    unseen += 1
                    break
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    successes += int(
                        terminated
                        and (
                            policy.env_id == "CliffWalking-v1"
                            # chain worlds: success = the full return of 1
                            or (policy.env_id in CHAIN_ENVS and total > 1 - 1e-9)
                            or (policy.env_id not in CHAIN_ENVS and reward > 0)
                        )
                    )
                    break
            returns.append(total)
            lengths.append(steps)
    finally:
        env.close()
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(lengths)),
        "success_rate": successes / episodes,
        "unseen_episodes": unseen,
    }
