"""Run one TD(lambda) configuration on one environment and seed.

``run`` returns a JSON-serialisable dict with the configuration, per-episode
results (return, steps, terminated, truncated), the fraction of truncated
episodes (required by D4) and, in predict mode on tabular features, the final
state values.

Seeding (D8): one base seed is split into independent streams for the
environment and for the policy, so changing one source of randomness never
shifts the other.
"""
from __future__ import annotations

import numpy as np

from td.adapters import make_adapter
from td.learner import LinearTD
from td.policies import epsilon_greedy, uniform_random


def run(
    env_id: str,
    *,
    mode: str,
    lam: float,
    alpha: float,
    gamma: float,
    episodes: int,
    seed: int,
    max_steps: int,
    epsilon: float = 0.1,
    update_rule: str = "true_online",
) -> dict:
    """Train for ``episodes`` episodes and return the results.

    Args:
        mode: "predict" (learn V of the uniform-random policy; the fixed
            policy is pending D2) or "control" (SARSA(lambda), epsilon-greedy).
        epsilon: Exploration rate, control mode only.
        update_rule: "true_online" (default) or "offline" (D3, under discussion).
    """
    if mode not in ("predict", "control"):
        raise ValueError("mode must be 'predict' or 'control'")
    adapter = make_adapter(env_id, max_steps=max_steps)
    learner = LinearTD(
        adapter.n_features,
        n_actions=adapter.n_actions if mode == "control" else None,
        alpha=alpha,
        gamma=gamma,
        lam=lam,
        update_rule=update_rule,
    )
    env_ss, policy_ss = np.random.SeedSequence(seed).spawn(2)
    env_seed = int(env_ss.generate_state(1)[0])
    rng = np.random.default_rng(policy_ss)

    def choose(phi) -> int:
        if mode == "control":
            return epsilon_greedy(learner.q_values(phi), epsilon, rng)
        return uniform_random(adapter.n_actions, rng)

    env = adapter.make()
    results = []
    try:
        for ep_idx in range(episodes):
            obs, _ = env.reset(seed=env_seed if ep_idx == 0 else None)
            phi = adapter.features(obs, env)
            action = choose(phi)
            learner.begin_episode(phi, action if mode == "control" else None)
            total, steps = 0.0, 0
            while True:
                obs, reward, terminated, truncated, _ = env.step(action)
                phi = adapter.features(obs, env)
                action = choose(phi)  # with true_online, uses the weights updated so far
                learner.observe(
                    float(reward),
                    phi,
                    action if mode == "control" else None,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            results.append(
                dict(
                    episode=ep_idx,
                    return_=total,
                    steps=steps,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
    finally:
        env.close()

    out = dict(
        env=env_id,
        mode=mode,
        update_rule=update_rule,
        lam=lam,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon if mode == "control" else None,
        episodes_run=episodes,
        seed=seed,
        max_steps=max_steps,
        truncation_fraction=sum(r["truncated"] for r in results) / episodes,
        episodes=results,
    )
    if mode == "predict":
        out["values"] = learner.weights.tolist()
    return out
