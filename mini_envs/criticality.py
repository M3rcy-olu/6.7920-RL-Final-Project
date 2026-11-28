"""Exact critical-state scores for the chain-of-rooms worlds (test 3).

State importance, after Karino, Ohmura & Kuniyoshi (2020), but with the
optimal Q* in place of the current-policy Q^pi::

    SI(s) = Var_{a ~ Uniform(A)} [ Q*(s, a) ]

Standard setup (design doc, Critical states): gamma = 0.99 and the end-only
reward (``feedback_every=None``). The chain worlds are deterministic and small
(about 1k to 9k reachable states), so Q* comes from exact value iteration on
the enumerated model rather than from a learner.

The action set is the env's own five actions (left, right, forward, pickup,
toggle), which is already the task action set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from mini_envs.chain_rooms import ChainRoomsEnv, _successors, plan, reachable_states

GAMMA = 0.99

# Percentile bands for test 3: name -> (low, high] on the percentile rank.
BANDS: dict[str, tuple[float, float]] = {
    "high": (0.90, 1.00),
    "mid": (0.40, 0.50),
    "low": (0.00, 0.10),
}


@dataclass
class Criticality:
    """Q*, SI, gap and percentile rank for every non-terminal state."""

    states: list[tuple]
    q: np.ndarray  # (n_states, n_actions)
    si: np.ndarray  # Var over actions of Q*
    gap: np.ndarray  # max_a Q* - min_a Q*
    rank: np.ndarray  # percentile rank of SI in (0, 1] among live states; NaN if lost
    gamma: float
    index: dict[tuple, int] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {s: i for i, s in enumerate(self.states)}

    def band(self, name: str) -> set[tuple]:
        lo, hi = BANDS[name]
        return {s for s, r in zip(self.states, self.rank) if lo < r <= hi}  # NaN (lost) never matches

    def log_band(self, name: str) -> set[tuple]:
        """Band on equal-width intervals of log10 SI over live states (the test-3 bands).

        Percentile bands follow rank only, so they ignore how far apart SI values
        are; equal-width log intervals keep a handful of large-SI decision states
        in "high" instead of filling it with near-tie step-cost states.
        """
        lo, hi = BANDS[name]
        z = self.log_position()
        return {s for s, v in zip(self.states, z) if (lo < v <= hi) or (lo == 0.0 and v == 0.0)}

    def log_position(self) -> np.ndarray:
        """Position of log10 SI in [0, 1] across live states; NaN if lost."""
        live = ~np.isnan(self.rank)
        out = np.full(len(self.states), np.nan)
        l = np.log10(np.maximum(self.si[live], 1e-300))
        span = l.max() - l.min()
        out[live] = (l - l.min()) / span if span > 0 else 1.0
        return out

    def greedy(self, state: tuple) -> int:
        return int(np.argmax(self.q[self.index[state]]))


def standard_env(env_id: str) -> ChainRoomsEnv:
    """The world in the standard setup: end-only reward."""
    return gym.make(env_id, feedback_every=None).unwrapped


def build_model(env: ChainRoomsEnv):
    """Deterministic model over non-terminal states: next index, reward, terminal flag."""
    states = sorted((s for s in reachable_states(env) if not _is_terminal(env, s)), key=repr)
    index = {s: i for i, s in enumerate(states)}
    n, n_a = len(states), env.action_space.n
    nxt = np.full((n, n_a), -1, dtype=np.int64)
    rew = np.zeros((n, n_a))
    term = np.zeros((n, n_a), dtype=bool)
    for i, s in enumerate(states):
        for a, s2, r, t in _successors(env, s):
            rew[i, a], term[i, a] = r, t
            if not t:
                nxt[i, a] = index[s2]
    env.reset(seed=0)
    return states, nxt, rew, term


def _is_terminal(env: ChainRoomsEnv, state: tuple) -> bool:
    return state[4] == env.n_rooms  # frontier reached the goal room


def value_iteration(nxt, rew, term, gamma: float = GAMMA, tol: float = 1e-12, max_iter: int = 100_000):
    n = nxt.shape[0]
    v = np.zeros(n)
    for _ in range(max_iter):
        cont = np.where(term, 0.0, v[np.clip(nxt, 0, None)])
        q = rew + gamma * cont
        v_new = q.max(axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            return q
        v = v_new
    raise RuntimeError("value iteration did not converge")


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Share of states with score <= x(s); ties share the same rank."""
    sorted_x = np.sort(x)
    return np.searchsorted(sorted_x, x, side="right") / len(x)


def compute(env: ChainRoomsEnv, gamma: float = GAMMA) -> Criticality:
    if env.n_gaps != 1:
        raise ValueError("criticality uses the standard setup: build the env with feedback_every=None")
    states, nxt, rew, term = build_model(env)
    q = value_iteration(nxt, rew, term, gamma)
    si = q.var(axis=1)
    gap = q.max(axis=1) - q.min(axis=1)
    # Round so float noise does not split true ties.
    # States after a wrong case-1 door have V* = 0 and SI = 0 for every action:
    # the outcome is already decided, so they are left out of the ranking.
    live = q.max(axis=1) > 0
    rank = np.full(len(states), np.nan)
    rank[live] = percentile_rank(np.round(si[live], 12))
    return Criticality(states, q, si, gap, rank, gamma)


def check_greedy_matches_plan(env: ChainRoomsEnv, crit: Criticality) -> None:
    """Greedy rollout under Q* must earn return 1 in exactly len(plan(env)) steps."""
    target = len(plan(env))
    env.reset(seed=0)
    total, steps = 0.0, 0
    while steps <= target:
        _, r, terminated, _, _ = env.step(crit.greedy(env.markov_state()))
        total, steps = total + r, steps + 1
        if terminated:
            break
    env.reset(seed=0)
    if not (terminated and steps == target and np.isclose(total, 1.0)):
        raise AssertionError(f"greedy Q* took {steps} steps (plan {target}), return {total}")
