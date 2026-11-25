"""Train and export a policy, or evaluate a previously saved policy."""

import argparse
import json
from pathlib import Path

from .adapters import ENVS
from .core import Policy, evaluate, train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--env", choices=ENVS, default=ENVS[0])
parser.add_argument("--episodes", type=int, default=20000)
parser.add_argument("--max-steps", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path)
parser.add_argument("--load", type=Path)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--initial-value", type=float, default=0.0)
parser.add_argument("--progress", action="store_true")
args = parser.parse_args()
if args.load:
    policy = Policy.load(args.load)
else:
    if args.output is None:
        parser.error("--output is required when training")
    policy = train(
        args.env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        initial_value=args.initial_value,
        progress=args.progress,
    )
    policy.save(args.output)
print(json.dumps(evaluate(policy), indent=2))
