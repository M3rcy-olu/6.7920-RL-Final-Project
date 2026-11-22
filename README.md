# 6.7920 RL bootstrap

Starter baseline for Nathaniel and Mercy's 6.7920 RL final project. Environment setup and TD(lambda) updates are separate modules.

## Reproduce

Tested on Python 3.13.14, Linux, CPU only. Exact package versions, including transitive dependencies, are in `requirements.txt`.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python rl_baseline.py --env FrozenLake-v1 --episodes 10 --seed 0
.venv/bin/python rl_baseline.py --env CliffWalking-v1 --episodes 10 --seed 0
```

No graphical dependencies are needed. Output is JSON with per-episode returns, transition counts, termination/truncation flags, parameters and estimated values. FrozenLake defaults to the standard slippery 4x4 map. CliffWalking uses the deterministic standard map. Both have an explicit 100-transition episode limit.

## What is implemented

- Seeded environment reset and independent seeded action RNG.
- Fixed uniform-random-policy tabular TD(lambda), online accumulating traces, zero initialization, alpha=0.1, gamma=0.99, lambda=0.8.
- Traces reset every episode. True termination has no bootstrap; time-limit truncation bootstraps the next state's value. This treats the cap as an external sampling cutoff, not a finite-horizon terminal state.
- Eight automated checks: repeatability for both environments, deterministic FrozenLake success path, cliff penalty/reset and safe path, time-limit semantics, one-step TD, terminal target, and trace reward propagation.

Verified locally: `8 passed` on 2026-11-18. These are correctness/smoke tests, not evidence of policy learning, convergence, or a project hypothesis. A random policy may never reach the FrozenLake goal during a short run.

## Scope and next steps

This is policy evaluation, not SARSA(lambda), Q(lambda), Q-learning or an optimal-control result. Online accumulating TD(1) with constant step size is not claimed equivalent to offline Monte Carlo. Agree on the shared TD interface with Mercy before integration.

Exact-reference validation, a full-return Monte Carlo comparator and matched-seed transition-budget comparisons are now implemented. See the findings below. Not implemented: MiniGrid, reward-delay/noise/decision wrappers, project hypothesis sweeps, plots or external paper reproduction. Current gamma=0.99 is a baseline setting only. Return-preserving reward redistribution or time dilation needs an explicit discounting decision before comparison. Hold other factors fixed and report both episode and transition budgets.

Upstream environment documentation: https://gymnasium.farama.org/environments/toy_text/frozen_lake/ and https://gymnasium.farama.org/environments/toy_text/cliff_walking/ .


## Exact-reference validation findings

[Full report and state-value grids](validation-results/report.md),
[exact references](validation-results/references.json),
[per-seed records](validation-results/runs.json),
[aggregate results](validation-results/summary.json).

Uniform-random policy, gamma 0.99, seeds 0 through 19, matched budgets of
1,000 / 10,000 / 100,000 transitions. All methods receive identical streams.
Exact start values: FrozenLake 0.012356137, CliffWalking -1072.236026683.
References were independently checked by iterative Bellman evaluation.

Final-budget RMSE, mean ± sample SD across seeds:

| Environment | TD(0) | TD(0.8) | Every-visit MC |
|---|---:|---:|---:|
| FrozenLake | 0.038205 ± 0.017926 | 0.055906 ± 0.026775 | 0.009382 ± 0.004459 |
| CliffWalking | 237.661455 ± 14.130311 | 118.234804 ± 26.938966 | 43.463285 ± 14.710777 |

RMSE evaluates reachable nonterminal states only. MC has the lowest final-budget
error in these settings; traces help CliffWalking but not FrozenLake at the final
checkpoint. This is not a general ranking or a tuned comparison: TD uses constant
alpha 0.1, while MC uses sample averages.

**Important:** validation removes the starter's 100-step cap for every method,
so full-return MC and TD target the same uncapped episodic value. Unfinished
episodes consume budget but give MC no update or free tail. CliffWalking has
only 7 to 23 completed episodes by 100k transitions. Original starter behavior
is unchanged. This is fixed-policy validation, not external implementation
reproduction or a test of the sparse-reward project hypothesis.

```sh
.venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 .venv/bin/python reference_validation.py
.venv/bin/python summarize_validation.py
```

The expanded suite has 12 tests. The saved sweep contains 360 records and 120
matched environment/seed/budget triplets. No hyperparameter tuning was performed.
