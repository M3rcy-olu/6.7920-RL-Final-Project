"""One shared TD(lambda) learner, linear in the state features phi(s).

Prediction learns V(s) = w . phi(s). Control learns Q(s, a) = w[a] . phi(s)
with Sarsa-style bootstrapping. Tabular learning is the special case where
phi(s) is one-hot (a single active index).

Feature formats accepted everywhere a feature vector is expected:

* an int, or an integer array: indices of binary features that are 1
  (one-hot or tile coding). The step size is split across the active
  features, so ``alpha`` means the same thing for any number of tilings (D6).
* a float array of length ``n_features``: a dense vector (plain gradient step).

Two update rules (D3, under discussion):

* ``"true_online"`` (default): true online TD(lambda) / Sarsa(lambda) with
  dutch traces (van Seijen & Sutton 2014; Sutton & Barto 2018, sections 12.5
  and 12.7). Weights change after every step. lam = 0 is classic online TD(0);
  lam = 1 ends a terminated episode with exactly the Monte Carlo weights.
* ``"offline"``: the offline lambda-return. Nothing changes during the
  episode; at its end, targets come from the start-of-episode weights and each
  step t = 0..T-1 moves toward its target in time order.

Either way the final state's value is used only when the episode was
truncated by the step limit (D4).

Per-step use (what the runner does)::

    learner.begin_episode(phi0, a0)
    learner.observe(r, phi1, a1, terminated=False, truncated=False)
    ...

or, for a finished episode, ``learner.update(Episode(...))``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from td.lambda_return import lambda_returns

UPDATE_RULES = ("true_online", "offline")


@dataclass
class Episode:
    """One finished episode.

    features: phi(S_0) .. phi(S_T), length T + 1 (includes the final state).
    rewards: R_1 .. R_T, length T.
    actions: A_0 .. A_T, length T + 1, required in control mode. A_T is the
        action the policy picked at S_T; it is only used to bootstrap a
        truncated episode.
    terminated: True for a real termination, False for a step-limit cutoff.
    """

    features: list
    rewards: list[float]
    terminated: bool
    actions: list[int] | None = field(default=None)

    def __post_init__(self):
        if len(self.features) != len(self.rewards) + 1:
            raise ValueError("need len(features) == len(rewards) + 1")
        if self.actions is not None and len(self.actions) != len(self.features):
            raise ValueError("need len(actions) == len(features)")


def _is_binary(phi) -> bool:
    return np.issubdtype(np.asarray(phi).dtype, np.integer)


class LinearTD:
    """TD(lambda) learner for prediction or Sarsa control.

    Args:
        n_features: Length of phi(s).
        n_actions: None for prediction (learn V); the action count for
            control (learn Q).
        alpha: Step size (per active feature for binary features).
        gamma: Discount factor.
        lam: Lambda in [0, 1]; 0 = TD(0), 1 = Monte Carlo.
        update_rule: "true_online" (default) or "offline".
    """

    def __init__(
        self,
        n_features: int,
        *,
        alpha: float,
        gamma: float,
        lam: float,
        n_actions: int | None = None,
        update_rule: str = "true_online",
    ):
        if not 0 < alpha or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
            raise ValueError("need alpha > 0, gamma in [0, 1], lam in [0, 1]")
        if update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha, self.gamma, self.lam = alpha, gamma, lam
        self.update_rule = update_rule
        shape = (n_features,) if n_actions is None else (n_actions, n_features)
        self.weights = np.zeros(shape)
        self._ep: dict | None = None

    @property
    def mode(self) -> str:
        return "predict" if self.n_actions is None else "control"

    # ---- estimates -------------------------------------------------------
    @staticmethod
    def _dot(w: np.ndarray, phi) -> float | np.ndarray:
        phi = np.asarray(phi)
        if _is_binary(phi):
            return w[..., np.atleast_1d(phi)].sum(axis=-1)
        return w @ phi

    def value(self, phi) -> float:
        if self.n_actions is not None:
            raise ValueError("value() is for predict mode; use q_values()")
        return float(self._dot(self.weights, phi))

    def q_values(self, phi) -> np.ndarray:
        if self.n_actions is None:
            raise ValueError("q_values() is for control mode; use value()")
        return np.asarray(self._dot(self.weights, phi), dtype=float)

    def _estimate(self, w: np.ndarray, phi, action) -> float:
        row = w if self.n_actions is None else w[action]
        return float(self._dot(row, phi))

    def _step_size(self, phi) -> float:
        return self.alpha / np.atleast_1d(phi).size if _is_binary(phi) else self.alpha

    def _flat_features(self, phi, action) -> np.ndarray:
        """phi(s) (predict) or the state-action vector x(s, a) (control), flattened like the weights."""
        x = np.zeros(self.weights.size)
        offset = 0 if self.n_actions is None else action * self.n_features
        block = x[offset : offset + self.n_features]
        if _is_binary(phi):
            np.add.at(block, np.atleast_1d(phi), 1.0)
        else:
            block += np.asarray(phi, dtype=float)
        return x

    def _check_action(self, action) -> None:
        if self.n_actions is not None and action is None:
            raise ValueError("control mode needs actions")

    # ---- per-step interface ----------------------------------------------
    def begin_episode(self, phi, action: int | None = None) -> None:
        self._check_action(action)
        if self.update_rule == "offline":
            self._ep = dict(features=[phi], actions=[action], rewards=[])
        else:
            self._ep = dict(
                x=self._flat_features(phi, action),
                alpha=self._step_size(phi),
                z=np.zeros(self.weights.size),
                v_old=0.0,
            )

    def observe(
        self,
        reward: float,
        phi_next,
        action_next: int | None = None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record one transition. The episode ends when terminated or truncated is True."""
        if self._ep is None:
            raise RuntimeError("call begin_episode() first")
        self._check_action(action_next)
        done = terminated or truncated

        if self.update_rule == "offline":
            ep = self._ep
            ep["features"].append(phi_next)
            ep["actions"].append(action_next)
            ep["rewards"].append(float(reward))
            if done:
                self._ep = None
                self._offline_update(
                    Episode(
                        features=ep["features"],
                        rewards=ep["rewards"],
                        terminated=terminated,
                        actions=ep["actions"] if self.n_actions is not None else None,
                    )
                )
            return

        # True online TD(lambda), Sutton & Barto (2018) section 12.5 (Sarsa: 12.7).
        ep, w = self._ep, self.weights.reshape(-1)
        x, a, z = ep["x"], ep["alpha"], ep["z"]
        x_next = np.zeros_like(x) if terminated else self._flat_features(phi_next, action_next)
        v, v_next = w @ x, w @ x_next
        delta = reward + self.gamma * v_next - v
        gl = self.gamma * self.lam
        zx = z @ x  # uses the trace before decay: z <- gl*z + (1 - a*gl*z.x) x
        z *= gl
        z += (1.0 - a * gl * zx) * x
        w += a * (delta + v - ep["v_old"]) * z - a * (v - ep["v_old"]) * x
        ep["v_old"], ep["x"] = v_next, x_next
        if not terminated:
            ep["alpha"] = self._step_size(phi_next)
        if done:
            self._ep = None

    # ---- whole-episode interface -----------------------------------------
    def update(self, ep: Episode) -> None:
        """Learn from one finished episode with the configured update rule."""
        self._check_action(None if ep.actions is None else 0)
        actions = ep.actions if ep.actions is not None else [None] * len(ep.features)
        if self.update_rule == "offline":
            self._offline_update(ep)
            return
        self.begin_episode(ep.features[0], actions[0])
        T = len(ep.rewards)
        for t in range(T):
            last = t == T - 1
            self.observe(
                ep.rewards[t],
                ep.features[t + 1],
                actions[t + 1],
                terminated=last and ep.terminated,
                truncated=last and not ep.terminated,
            )

    def _offline_update(self, ep: Episode) -> np.ndarray:
        actions = ep.actions if ep.actions is not None else [None] * len(ep.features)
        start = self.weights.copy()
        next_values = [
            self._estimate(start, ep.features[t + 1], actions[t + 1])
            for t in range(len(ep.rewards))
        ]
        targets = lambda_returns(
            ep.rewards, next_values, terminated=ep.terminated, gamma=self.gamma, lam=self.lam
        )
        for t, target in enumerate(targets):
            phi, a = ep.features[t], actions[t]
            row = self.weights if self.n_actions is None else self.weights[a]
            error = target - self._estimate(self.weights, phi, a)
            if _is_binary(phi):
                idx = np.atleast_1d(phi)
                np.add.at(row, idx, self.alpha / idx.size * error)
            else:
                row += self.alpha * error * np.asarray(phi, dtype=float)
        return targets
