"""Sweep TD(lambda) over lambda x alpha x seeds on one environment.

Examples (from the repo root):

    python scripts/run_td.py --env FrozenLake-v1 --mode predict --lam 0 0.5 1 --alpha 0.01 0.05
    python scripts/run_td.py --env FrozenLake-v1 --mode predict --update offline --lam 0 1
    python scripts/run_td.py --env CliffWalking-v1 --mode control --lam 0 0.9 1 \
        --alpha 0.1 0.5 --seeds 5 --episodes 500 --out outputs/cliff.jsonl

Each (lambda, alpha, seed) run is one JSON line in ``--out`` (default:
outputs/td_<env>_<mode>_<update>.jsonl) with per-episode results and the truncation
fraction. A summary table (mean over seeds) is printed at the end. Choosing
the best alpha per lambda (D6) is left to analysis, because the metric (D7)
is not decided yet.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from td.runner import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True)
    p.add_argument("--mode", choices=["predict", "control"], required=True)
    p.add_argument("--update", choices=["true_online", "offline"], default="true_online",
                   help="update rule (D3, under discussion)")
    p.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.5, 0.9, 1.0])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.1])
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=0.1, help="control mode only")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3, help="number of seeds, 0..N-1")
    p.add_argument("--seed-offset", type=int, default=0, help="first seed (keep tuning and final seeds apart, D8)")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = Path(args.out or f"outputs/td_{args.env}_{args.mode}_{args.update}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed_offset, args.seed_offset + args.seeds)

    summary = {}
    with out.open("w") as fh:
        for lam, alpha, seed in itertools.product(args.lam, args.alpha, seeds):
            res = run(args.env, mode=args.mode, lam=lam, alpha=alpha, gamma=args.gamma,
                      episodes=args.episodes, seed=seed, max_steps=args.max_steps,
                      epsilon=args.epsilon, update_rule=args.update)
            fh.write(json.dumps(res) + "\n")
            tail = res["episodes"][-max(1, args.episodes // 10):]
            summary.setdefault((lam, alpha), []).append(
                (np.mean([e["return_"] for e in tail]), res["truncation_fraction"])
            )

    print(f"wrote {out}")
    print(f"{'lambda':>7} {'alpha':>7} {'return (last 10%)':>18} {'truncated':>10}")
    for (lam, alpha), rows in summary.items():
        ret, trunc = np.mean(rows, axis=0)
        print(f"{lam:>7.2f} {alpha:>7.3f} {ret:>18.3f} {trunc:>10.1%}")


if __name__ == "__main__":
    main()
