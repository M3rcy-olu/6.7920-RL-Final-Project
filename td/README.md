# td: shared TD(λ) learner

One learner, linear in state features φ(s), for λ from 0 (TD(0)) to 1 (Monte Carlo). It
implements the decisions in the experiment-design doc: D4 bootstrap at the step-limit cutoff, D6 α
split across active features, D8 separate seeds per randomness source.

Two update rules (D3, under discussion), chosen with `update_rule=` / `--update`:

- `true_online` (default): true online TD(λ) / Sarsa(λ) with dutch traces. Updates every step.
  λ=0 is classic online TD(0); λ=1 ends a terminated episode with exactly the Monte Carlo weights.
  Both equivalences are unit-tested in `tests/test_true_online.py`.
- `offline`: the offline λ-return. Updates only at episode end, from start-of-episode values.

| file | what it does |
|---|---|
| `lambda_return.py` | `lambda_returns(rewards, next_values, terminated, gamma, lam)`: backward λ-return targets (offline rule) |
| `learner.py` | `LinearTD` (predict: V = w·φ, control: Q = w[a]·φ, Sarsa), per-step `begin_episode` / `observe`, whole-episode `update(Episode)` |
| `policies.py` | `epsilon_greedy`, `uniform_random` |
| `adapters.py` | per-env build + φ encoder + sizes; toy-text only so far (Minigrid/Miniworld pending D11) |
| `runner.py` | `run(env_id, mode, lam, alpha, gamma, episodes, seed, max_steps, epsilon, update_rule)` → JSON-able dict |

```sh
python scripts/run_td.py --env CliffWalking-v1 --mode control --lam 0 0.5 0.9 1 --alpha 0.1 0.5 --seeds 5
python scripts/run_td.py --env FrozenLake-v1 --mode predict --update offline --lam 0 1 --alpha 0.01 --episodes 5000
```

Features φ can be an int or an int array (indices of active binary features: one-hot or tile
coding) or a dense float array. Tabular learning is φ = one-hot.

Predict mode evaluates the uniform-random policy until the fixed policy is chosen (D2).
`tests/test_runner.py` checks predict mode, under both update rules, against exact
dynamic-programming values on FrozenLake.

Check `truncation_fraction` in every result (D4). On CliffWalking control with a 100-step limit
(300 episodes, 3 seeds, α=0.1), `offline` truncated 34–50% of episodes for λ<1 and 99.8% at λ=1;
`true_online` truncated 8–12% for λ<1 but still 83% at λ=1. That task needs a longer limit.
