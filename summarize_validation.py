"""Build human-readable report from saved, verified experiment outputs."""
import json
from pathlib import Path
import numpy as np

p = Path('validation-results')
refs = json.loads((p/'references.json').read_text())
rows = json.loads((p/'runs.json').read_text())
summary = json.loads((p/'summary.json').read_text())
assert len(rows) == 2*20*3*3
assert all(np.isfinite(r['rmse']) for r in rows)
lines = ['# Exact-reference RL validation', '',
         'Computed locally on CPU, 2026-11-21. This is fixed-policy evaluation, not control or external-paper reproduction.', '',
         '## Protocol', '',
         '- Environments: Gymnasium slippery 4x4 FrozenLake-v1 and deterministic CliffWalking-v1.',
         '- Policy: uniform random. Gamma 0.99. All estimates initialized to zero.',
         '- TD(0) and accumulating TD(0.8): constant alpha 0.1. MC: every-visit sample-average full discounted returns, no oracle bootstrapping.',
         '- Seeds 0 through 19, prespecified budgets 1,000, 10,000, 100,000 transitions. Each seed uses the exact same transition stream for all three methods, independent seeded action RNG and reset seed + episode index.',
         '- Important change from starter: the 100-step TimeLimit is removed for ALL methods. The target is the uncapped episodic value. This permits true full-return MC without treating truncations as terminal or using exact values inside the estimator.',
         '- At a checkpoint, MC uses only completed episodes. An unfinished episode gets no MC update and no free rollout tail. TD can use its transitions immediately. Episode counts, pending transitions and state coverage are logged.',
         '- Reference: solve (I - gamma P_nonterminal) V = r from environment transition model under uniform actions. Independently checked with iterative Bellman evaluation from raw transition tuples.',
         '- Metric: unweighted RMSE over reset-reachable nonterminal states only. Excludes FrozenLake holes/goal and unreachable CliffWalking cliff cells/terminal goal.',
         '- No hyperparameter tuning, no separate external implementation, no hypothesis or reward-sparsity experiment. MC and TD use different estimator/step-size conventions, so this is not an algorithm-isolation claim.', '',
         '## References', '']
for name, ref in refs.items():
    lines += [f"### {name}", f"Start value: {ref['start_value']:.9f}. Zero-init RMSE: {ref['zero_init_rmse']:.9f}. Bellman residual: {ref['bellman_residual']:.3g}.",
              f"Evaluated state IDs: {ref['evaluated_states']}", '', 'State-value grid (NaN marks terminal or unreachable states excluded from evaluation):', '```',
              np.array2string(np.array([v if i in ref['evaluated_states'] else np.nan for i, v in enumerate(ref['values'])]).reshape((4,4) if 'Frozen' in name else (4,12)), precision=6), '```', '']
lines += ['## Results', '', 'Mean RMSE ± sample standard deviation across 20 seeds. SD is seed variation, not a confidence interval.', '',
          '| Environment | Transitions | Method | RMSE mean | SD | Completed episodes range |',
          '|---|---:|---|---:|---:|---|']
for r in summary:
    lines.append(f"| {r['environment']} | {r['transitions']:,} | {r['method']} | {r['rmse_mean']:.6f} | {r['rmse_sd']:.6f} | {r['completed_episodes_range']} |")
lines += ['', '## Paired final-budget differences', '',
          'Mean (method RMSE minus TD(0) RMSE), with approximate 95% t intervals (19 df, t=2.093). Negative favors the method. These are exploratory, not multiplicity-corrected.', '']
for name in refs:
    for method in ('td08', 'mc'):
        errors = {m: {r['seed']: r['rmse'] for r in rows if r['environment']==name and r['transitions']==100000 and r['method']==m} for m in ('td0', method)}
        diff = np.array([errors[method][s]-errors['td0'][s] for s in range(20)])
        margin = 2.093*diff.std(ddof=1)/np.sqrt(20)
        lines.append(f"- {name}, {method}: {diff.mean():.6f}, interval [{diff.mean()-margin:.6f}, {diff.mean()+margin:.6f}].")
lines += ['', '## Reproduce', '', '```sh', 'cd /path/to/67920-rl-bootstrap', '.venv/bin/python -m pytest -q',
          'OPENBLAS_NUM_THREADS=1 .venv/bin/python reference_validation.py', '.venv/bin/python summarize_validation.py', '```', '',
          'Pinned package versions: requirements.txt. Machine-readable exact values: validation-results/references.json. Per-seed checkpoint records: runs.json. Aggregate results: summary.json.', '',
          '## Interpretation limits', '',
          'Exact DP validates the target and tests verify update semantics and deterministic pairing. Finite-budget RMSE is empirical, not a convergence guarantee. Constant-step-size TD retains variance. Full-return MC can be delayed by long episodes, especially CliffWalking, and incomplete returns at checkpoints create finite-budget censoring. Outcomes do not establish policy improvement or reproduce the proposed sparse-reward project. No code pushed or PR modified.', '']
(p/'report.md').write_text('\n'.join(lines))
print('\n'.join(lines))
