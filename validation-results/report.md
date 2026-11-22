# Exact-reference RL validation

Computed locally on CPU, 2026-11-21. This is fixed-policy evaluation, not control or external-paper reproduction.

## Protocol

- Environments: Gymnasium slippery 4x4 FrozenLake-v1 and deterministic CliffWalking-v1.
- Policy: uniform random. Gamma 0.99. All estimates initialized to zero.
- TD(0) and accumulating TD(0.8): constant alpha 0.1. MC: every-visit sample-average full discounted returns, no oracle bootstrapping.
- Seeds 0 through 19, prespecified budgets 1,000, 10,000, 100,000 transitions. Each seed uses the exact same transition stream for all three methods, independent seeded action RNG and reset seed + episode index.
- Important change from starter: the 100-step TimeLimit is removed for ALL methods. The target is the uncapped episodic value. This permits true full-return MC without treating truncations as terminal or using exact values inside the estimator.
- At a checkpoint, MC uses only completed episodes. An unfinished episode gets no MC update and no free rollout tail. TD can use its transitions immediately. Episode counts, pending transitions and state coverage are logged.
- Reference: solve (I - gamma P_nonterminal) V = r from environment transition model under uniform actions. Independently checked with iterative Bellman evaluation from raw transition tuples.
- Metric: unweighted RMSE over reset-reachable nonterminal states only. Excludes FrozenLake holes/goal and unreachable CliffWalking cliff cells/terminal goal.
- No hyperparameter tuning, no separate external implementation, no hypothesis or reward-sparsity experiment. MC and TD use different estimator/step-size conventions, so this is not an algorithm-isolation claim.

## References

### FrozenLake-v1
Start value: 0.012356137. Zero-init RMSE: 0.149736940. Bellman residual: 6.94e-18.
Evaluated state IDs: [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]

State-value grid (NaN marks terminal or unreachable states excluded from evaluation):
```
[[0.012356 0.010424 0.019338 0.009478]
 [0.014787      nan 0.038894      nan]
 [0.032602 0.084338 0.137811      nan]
 [     nan 0.170345 0.433579      nan]]
```

### CliffWalking-v1
Start value: -1072.236026683. Zero-init RMSE: 927.732044620. Bellman residual: 5.68e-13.
Evaluated state IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

State-value grid (NaN marks terminal or unreachable states excluded from evaluation):
```
[[ -929.137751  -933.75701   -937.752011  -938.905323  -936.297586
   -928.940179  -915.100611  -891.940495  -855.513986  -802.001147
   -732.816083  -668.674482]
 [ -958.019008  -968.068418  -974.442267  -976.561536  -974.837058
   -968.914874  -957.354515  -937.204485  -903.126134  -846.037053
   -753.340948  -627.509628]
 [-1011.51829  -1041.128858 -1050.718105 -1053.478122 -1052.921189
  -1049.635414 -1042.838676 -1030.223241 -1006.198553  -955.822893
   -833.398643  -481.826973]
 [-1072.236027          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan]]
```

## Results

Mean RMSE ± sample standard deviation across 20 seeds. SD is seed variation, not a confidence interval.

| Environment | Transitions | Method | RMSE mean | SD | Completed episodes range |
|---|---:|---|---:|---:|---|
| FrozenLake-v1 | 1,000 | td0 | 0.111857 | 0.019402 | [125, 145] |
| FrozenLake-v1 | 1,000 | td08 | 0.083961 | 0.033716 | [125, 145] |
| FrozenLake-v1 | 1,000 | mc | 0.096191 | 0.049652 | [125, 145] |
| FrozenLake-v1 | 10,000 | td0 | 0.033717 | 0.014837 | [1236, 1368] |
| FrozenLake-v1 | 10,000 | td08 | 0.049933 | 0.019987 | [1236, 1368] |
| FrozenLake-v1 | 10,000 | mc | 0.033065 | 0.012023 | [1236, 1368] |
| FrozenLake-v1 | 100,000 | td0 | 0.038205 | 0.017926 | [12874, 13165] |
| FrozenLake-v1 | 100,000 | td08 | 0.055906 | 0.026775 | [12874, 13165] |
| FrozenLake-v1 | 100,000 | mc | 0.009382 | 0.004459 | [12874, 13165] |
| CliffWalking-v1 | 1,000 | td0 | 901.501074 | 3.330962 | [0, 1] |
| CliffWalking-v1 | 1,000 | td08 | 837.803589 | 10.005619 | [0, 1] |
| CliffWalking-v1 | 1,000 | mc | 908.692595 | 85.147008 | [0, 1] |
| CliffWalking-v1 | 10,000 | td0 | 732.446089 | 8.823740 | [0, 4] |
| CliffWalking-v1 | 10,000 | td08 | 482.965434 | 23.610589 | [0, 4] |
| CliffWalking-v1 | 10,000 | mc | 480.026122 | 380.046952 | [0, 4] |
| CliffWalking-v1 | 100,000 | td0 | 237.661455 | 14.130311 | [7, 23] |
| CliffWalking-v1 | 100,000 | td08 | 118.234804 | 26.938966 | [7, 23] |
| CliffWalking-v1 | 100,000 | mc | 43.463285 | 14.710777 | [7, 23] |

## Paired final-budget differences

Mean (method RMSE minus TD(0) RMSE), with approximate 95% t intervals (19 df, t=2.093). Negative favors the method. These are exploratory, not multiplicity-corrected.

- FrozenLake-v1, td08: 0.017701, interval [0.008915, 0.026487].
- FrozenLake-v1, mc: -0.028823, interval [-0.037507, -0.020139].
- CliffWalking-v1, td08: -119.426651, interval [-130.263981, -108.589320].
- CliffWalking-v1, mc: -194.198169, interval [-202.195585, -186.200753].

## Reproduce

```sh
cd /path/to/67920-rl-bootstrap
.venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 .venv/bin/python reference_validation.py
.venv/bin/python summarize_validation.py
```

Pinned package versions: requirements.txt. Machine-readable exact values: validation-results/references.json. Per-seed checkpoint records: runs.json. Aggregate results: summary.json.

## Interpretation limits

Exact DP validates the target and tests verify update semantics and deterministic pairing. Finite-budget RMSE is empirical, not a convergence guarantee. Constant-step-size TD retains variance. Full-return MC can be delayed by long episodes, especially CliffWalking, and incomplete returns at checkpoints create finite-budget censoring. Outcomes do not establish policy improvement or reproduce the proposed sparse-reward project. No code pushed or PR modified.
