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

Not implemented: Monte Carlo comparator, exact DP reference values, MiniGrid, reward-delay/noise/decision wrappers, experiment sweeps, plots or paper results. Current gamma=0.99 is a baseline setting only. Return-preserving reward redistribution or time dilation needs an explicit discounting decision before comparison. Hold other factors fixed and report both episode and transition budgets.

Upstream environment documentation: https://gymnasium.farama.org/environments/toy_text/frozen_lake/ and https://gymnasium.farama.org/environments/toy_text/cliff_walking/ .
