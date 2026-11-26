# Saved policy provenance

Empirical tabular baselines, not optimality certificates. Evaluation uses frozen
greedy actions for 100 episodes, with reset seeds 100000 through 100099.
Training uses reset seed 0 + episode and gamma 0.99. Alpha is 0.2 for Gym
and 1 for KeyDoor. Epsilon decreases
linearly from 1 to 0.05 over the first 80% of episodes. The legacy
baselines below use 256-step horizons.

| Environment | Episodes | Initial Q | Success rate | Mean return | Mean steps | Unseen episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CliffWalking-v1 | 10000 | 0 | 1.00 | -13.00 | 13.00 | 0 |
| FrozenLake-v1, slippery 4x4 | 40000 | 0 | 0.77 | 0.77 | 47.14 | 0 |

KeyDoor's 30000-episode run with alpha=1 and initial Q=0.1 achieved 100/100
frozen evaluation successes, mean return 0.94375 and 16 steps, with no unseen
episodes. It collected 4291 goal rewards during exploration. Earlier runs with
alpha=0.2 failed frozen evaluation, including 10000 episodes with initial Q=0.1.
Training uses standard online one-step Q-learning, without replay,
demonstrations or reward shaping.

KeyDoor reset seeds do not generate held-out layouts: this task has a fixed
layout. Reward depends on elapsed steps, so state includes time and truncation
is terminal. Gym truncations bootstrap. See the parent README for the schema
and coverage limitations.

Tested with Python 3.13.14, gymnasium 1.4.0, minigrid 3.2.0, numpy 2.3.5,
miniworld 2.1.0, pyglet 1.5.31 and pytest 9.0.3. The existing mini_envs package
imports MiniWorld eagerly; Debian needs libglu1-mesa for headless MiniGrid runs.

## Chain study baselines

13 of 14 study worlds passed, based on `chain_worlds` commit e254c0a.
Cost-Graded-8 failed and has no saved policy. The four Matter worlds are
excluded. Training is ordinary online Q-learning, with no algorithm changes,
replay, demonstrations or reward shaping.

Shared selection rule:

- Horizon: 50 times each world's optimal path length, computed by the existing
  `mini_envs.chain_rooms.plan` helper. This leaves room for early random walks
  while scaling with task length. Cutoffs bootstrap, not terminal failures.
- Alpha 0.2, gamma 0.99 (matching the experiment defaults), initial Q 0.
- Unchanged epsilon schedule: `max(0.05, 1 - episode / (N * 0.8))`.
- Training seed 0, reset seeds `0 + episode`. Start with 2000 episodes;
  if needed, retrain from scratch at 4000, 8000, then 16000 episodes using
  the same seed. Keep the first passing budget, not the best of multiple seeds.
- Frozen greedy evaluation: 100 episodes, reset seeds 100000 through 100099.
  Require 100% full-return success, zero unseen-state episodes, and mean
  steps at most 1.10 times the optimal path. These are fixed layouts, so
  evaluation is not evidence of generalization to new layouts.
- Save and reload each policy, require identical actions and evaluation.
  The six-integer state encoder excludes accumulated loss so the frozen
  policies can be reused under changed feedback. No feedback experiments
  are run here.

Reproduce all runs: `python scripts/train_chain_baselines.py` from the repo
root. This overwrites the chain artifacts. Re-check one artifact with
`python -m q_learning --load q_learning/policies/chain_base.json`.
`chain_results.json` records each attempt and its elapsed training seconds
(excluding evaluation and serialization). Times are machine/load dependent.
A preliminary 1000-episode Base pilot passed but is not a candidate in this
shared selection rule and is not included in the table.


| World | Success | Mean steps | Optimal steps | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| Delay-L1 | 100% | 44 | 44 | 12.29 |
| Base | 100% | 56 | 56 | 15.18 |
| Delay-L8 | 100% | 72 | 72 | 18.89 |
| Delay-L16 | 100% | 104 | 104 | 30.53 |
| Doors-3 | 100% | 76 | 76 | 19.97 |
| Doors-4 | 100% | 89 | 89 | 34.87 |
| Rooms-2 | 100% | 36 | 36 | 8.98 |
| Rooms-6 | 100% | 106 | 106 | 26.77 |
| Rooms-8 | 100% | 132 | 130 | 36.34 |
| Size-7 | 100% | 65 | 65 | 21.57 |
| Hard | 100% | 150 | 144 | 64.98 |
| Cost-Graded | 100% | 56 | 56 | 14.97 |
| Cost-Graded-Rev | 100% | 56 | 56 | 15.23 |
| Cost-Graded-8 | 0% | 136 | 148 | 505.01 |

All 13 saved policies passed at 2000 episodes with mean return 1 and no
unseen-state episodes. Cost-Graded-8 failed at 2000, 4000, 8000 and 16000
episodes, with mean returns 0.85, 0.86875, 0.95625 and 0.95625 respectively.
The table shows its last completed evaluation and cumulative training time.
A 32000-episode retry was stopped before completion and is not included.

Diagnosis: exact value iteration on the existing transition model at gamma
0.99 also selects a 136-step route with return 0.95625 instead of the
148-step full-return route. Discounting favors sacrificing a small amount of
reward to finish sooner, so more training alone cannot meet this quality bar.
At gamma 0.999 the exact greedy route earns full return in 148 steps. This is
a planning diagnostic, not a trained baseline at 0.999. No discount or reward
changes were made. Whether to omit this world is pending Mercy's input.
