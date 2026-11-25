"""Step-limit test for the chain worlds (decisions D4 and D12).

For each world, feedback setting, lambda and step limit, this trains Sarsa(lambda)
(the shared LinearTD learner, accumulating traces, one-hot Markov state) and
reports the fraction of episodes cut off by the step limit. D4 asks for at most
5% cut off. Limits are given as multiples of the world's optimal path length L*.

    python scripts/chain_step_limits.py                         # all 15 worlds
    python scripts/chain_step_limits.py --worlds Base Hard --mults 10 20 50

The worlds are compiled to their exact transition table first, so this is
much faster than stepping the env (same dynamics; tested in test_chain_rooms).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import mini_envs  # noqa: E402,F401
from mini_envs.chain_rooms import plan, transition_table  # noqa: E402
from td.learner import LinearTD  # noqa: E402
from td.policies import epsilon_greedy  # noqa: E402

PREFIX, SUFFIX = "MiniGrid-Proj-Chain-", "-v0"


def sarsa_on_table(table, *, lam, limit, episodes, alpha, gamma, epsilon, seed):
    """Run Sarsa(lambda) on the table; return per-episode (return, cut off, steps)."""
    rng = np.random.default_rng(seed)
    n_actions = table.next.shape[1]
    learner = LinearTD(table.n_states, n_actions=n_actions, alpha=alpha, gamma=gamma, lam=lam)
    out = np.zeros((episodes, 3))
    for ep in range(episodes):
        s = table.start
        a = epsilon_greedy(learner.q_values(s), epsilon, rng)
        learner.begin_episode(s, a)
        ret, t = 0.0, 0
        while True:
            r, term, s2 = table.reward[s, a], bool(table.terminal[s, a]), int(table.next[s, a])
            t += 1
            ret += r
            trunc = not term and t >= limit
            a2 = epsilon_greedy(learner.q_values(s2), epsilon, rng)
            learner.observe(r, s2, a2, terminated=term, truncated=trunc)
            s, a = s2, a2
            if term or trunc:
                break
        out[ep] = (ret, trunc, t)
    return out


def one_world(job):
    name, fb, lams, mults, args = job
    env = gym.make(PREFIX + name + SUFFIX, feedback_every=fb).unwrapped
    l_star = len(plan(env))
    table = transition_table(env)
    rows = []
    for lam in lams:
        for m in mults:
            t0 = time.time()
            res = [
                sarsa_on_table(table, lam=lam, limit=m * l_star, episodes=args.episodes, alpha=args.alpha,
                               gamma=args.gamma, epsilon=args.epsilon, seed=1000 * seed)
                for seed in range(args.seeds)
            ]
            res = np.stack(res)  # seeds x episodes x 3
            last = res[:, -100:]
            rows.append(dict(
                world=name, feedback="end" if fb is None else fb, lam=lam, mult=m, limit=m * l_star,
                l_star=l_star, states=table.n_states,
                cut_all=float(res[:, :, 1].mean()), cut_first100=float(res[:, :100, 1].mean()),
                cut_last100=float(last[:, :, 1].mean()), full_last100=float((last[:, :, 0] > 0.999).mean()),
                ret_last100=float(last[:, :, 0].mean()), steps_last100=float(last[:, :, 2].mean()),
                secs=round(time.time() - t0, 1),
            ))
            print(f"{name:<12} fb={rows[-1]['feedback']!s:<4} lam={lam:<4} limit={m:>3}xL*={m * l_star:>6}"
                  f"  cut all {rows[-1]['cut_all']:6.1%}  last100 {rows[-1]['cut_last100']:6.1%}"
                  f"  full-return last100 {rows[-1]['full_last100']:6.1%}  ({rows[-1]['secs']}s)", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worlds", nargs="*", default=None, help="short names, e.g. Base Hard (default: all)")
    p.add_argument("--feedback", nargs="*", default=["1", "end"])
    p.add_argument("--lams", nargs="*", type=float, default=[0.0, 0.9, 1.0])
    p.add_argument("--mults", nargs="*", type=int, default=[10, 20, 50, 100])
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="outputs/chain_step_limits.csv")
    args = p.parse_args()

    from mini_envs.chain_worlds import CHAIN_WORLDS

    names = args.worlds or [k[len(PREFIX):-len(SUFFIX)] for k in CHAIN_WORLDS]
    fbs = [None if f == "end" else int(f) for f in args.feedback]
    jobs = [(n, fb, [lam], args.mults, args) for n in names for fb in fbs for lam in args.lams]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(one_world, jobs):
            rows.extend(r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
