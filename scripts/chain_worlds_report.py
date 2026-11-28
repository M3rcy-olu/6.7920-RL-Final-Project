"""Summarise the 15 chain-of-rooms worlds: size of the state space, optimal path length,
and how a uniform-random policy does (the calibration check in the decision doc).

    python scripts/chain_worlds_report.py                  # all worlds, 200 random episodes each
    python scripts/chain_worlds_report.py --episodes 50 --render outputs/chain_worlds

``--render DIR`` also saves a PNG of every world's start state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from mini_envs.chain_rooms import plan, reachable_states  # noqa: E402
from mini_envs.chain_worlds import CHAIN_WORLDS  # noqa: E402


def random_rollouts(env, episodes: int, seed: int) -> tuple[float, float, float]:
    """Return (fraction reaching the end, mean return, fraction cut off by the step limit)."""
    rng = np.random.default_rng(seed)
    env._skip_obs = True
    reached = cut = 0
    total = 0.0
    try:
        for ep in range(episodes):
            env.reset(seed=seed + ep)
            ret = 0.0
            while True:
                _, r, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
                ret += r
                if terminated or truncated:
                    reached += bool(terminated)
                    cut += bool(truncated)
                    break
            total += ret
    finally:
        env._skip_obs = False
    return reached / episodes, total / episodes, cut / episodes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", default=None, metavar="DIR")
    args = p.parse_args()

    header = f"{'world':<34}{'rooms':>6}{'c1':>4}{'size':>5}{'b':>3}{'L':>4}{'states':>8}{'L*':>5}{'limit':>7}{'rand end':>9}{'rand ret':>9}"
    print(header)
    print("-" * len(header))
    for env_id, spec in CHAIN_WORLDS.items():
        env = gym.make(env_id, render_mode="rgb_array" if args.render else None).unwrapped
        k = spec["kwargs"]
        n_states = len(reachable_states(env))
        l_star = len(plan(env))
        reach, ret, _ = random_rollouts(env, args.episodes, args.seed)
        short = env_id.replace("MiniGrid-Proj-Chain-", "").replace("-v0", "")
        print(
            f"{short:<34}{len(k['rooms']):>6}{k['rooms'].count('c1'):>4}{k['room_size']:>5}{k['doors']:>3}"
            f"{k['corridor_len']:>4}{n_states:>8}{l_star:>5}{env.max_steps:>7}{reach:>9.1%}{ret:>9.3f}"
        )
        if args.render:
            from PIL import Image

            out = Path(args.render)
            out.mkdir(parents=True, exist_ok=True)
            env.reset(seed=0)
            Image.fromarray(env.render()).save(out / f"{short}.png")
        env.close()


if __name__ == "__main__":
    main()
