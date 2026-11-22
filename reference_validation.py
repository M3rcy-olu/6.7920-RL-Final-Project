"""Exact uniform-policy references and paired transition-budget validation.

Use uncapped episodes for every method so full-return MC targets the same
infinite-horizon episodic value as TD. At checkpoints MC uses completed
returns only. Incomplete final episodes receive no MC updates or free tail.
"""
import json
from pathlib import Path
import numpy as np
from environments import make_env
from td_lambda import td_update

GAMMA = .99
BUDGETS = (1000, 10000, 100000)


def exact_reference(env, gamma=GAMMA):
    n = env.observation_space.n
    p, r = np.zeros((n, n)), np.zeros(n)
    for s, actions in env.P.items():
        for outcomes in actions.values():
            for prob, nxt, reward, terminal in outcomes:
                r[s] += prob * reward / env.action_space.n
                if not terminal:
                    p[s, nxt] += prob / env.action_space.n
    v = np.linalg.solve(np.eye(n) - gamma * p, r)
    # Independent iterative Bellman evaluation from raw transitions.
    w = np.zeros(n)
    for _ in range(10000):
        nxt_v = np.array([sum(prob * (rew + (0 if done else gamma * w[ns]))
                            / env.action_space.n
                            for outcomes in env.P[s].values()
                            for prob, ns, rew, done in outcomes) for s in range(n)])
        if np.max(np.abs(nxt_v - w)) < 1e-11:
            w = nxt_v
            break
        w = nxt_v
    assert np.max(np.abs(w-v)) < 2e-9
    # Reachable nonterminal states from reset support under uniform policy.
    reachable = set(np.flatnonzero(env.initial_state_distrib))
    frontier = list(reachable)
    while frontier:
        s = frontier.pop()
        for outcomes in env.P[s].values():
            for prob, ns, _, done in outcomes:
                if prob > 0 and not done and ns not in reachable:
                    reachable.add(ns)
                    frontier.append(ns)
    return v, sorted(int(s) for s in reachable), float(np.max(np.abs(v-r-gamma*p@v)))


def mc_update(values, counts, trajectory, gamma=GAMMA):
    """Every-visit sample-average MC, update only after true termination."""
    ret = 0.
    for state, reward in reversed(trajectory):
        ret = reward + gamma * ret
        counts[state] += 1
        values[state] += (ret-values[state])/counts[state]


def compare(name, seed, reference, mask, budgets=BUDGETS):
    wrapped = make_env(name)
    env = wrapped.unwrapped  # Intentional removal of 100-step sampling cap.
    rng = np.random.default_rng(seed)
    n = env.observation_space.n
    values = {k: np.zeros(n) for k in ('td0', 'td08', 'mc')}
    traces = {k: np.zeros(n) for k in ('td0', 'td08')}
    counts, visits = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    episode, trajectory, rows = 0, [], []
    state, _ = env.reset(seed=seed)
    for step in range(1, max(budgets)+1):
        action = int(rng.integers(env.action_space.n))
        ns, reward, done, trunc, _ = env.step(action)
        assert not trunc
        visits[state] += 1
        trajectory.append((state, reward))
        for method, lam in [('td0', 0), ('td08', .8)]:
            td_update(values[method], traces[method], state, reward, ns, done,
                      alpha=.1, gamma=GAMMA, lam=lam)
        if done:
            mc_update(values['mc'], counts, trajectory)
            episode += 1
            trajectory = []
            for z in traces.values():
                z.fill(0)
            ns, _ = env.reset(seed=seed+episode)
        state = ns
        if step in budgets:
            for method, v in values.items():
                rows.append(dict(environment=name, seed=seed, transitions=step,
                                 method=method, completed_episodes=episode,
                                 pending_mc_transitions=len(trajectory),
                                 visited_states=int(np.count_nonzero(visits[mask])),
                                 mc_updated_states=int(np.count_nonzero(counts[mask])),
                                 rmse=float(np.sqrt(np.mean((v[mask]-reference[mask])**2))),
                                 start_value=float(v[np.argmax(env.initial_state_distrib)])))
    wrapped.close()
    return rows


def main():
    out = Path('validation-results')
    out.mkdir(exist_ok=True)
    refs, rows = {}, []
    for name in ('FrozenLake-v1', 'CliffWalking-v1'):
        env = make_env(name)
        v, mask, residual = exact_reference(env.unwrapped)
        refs[name] = dict(values=v.tolist(), evaluated_states=mask,
                          bellman_residual=residual,
                          start_value=float(v[np.argmax(env.unwrapped.initial_state_distrib)]),
                          zero_init_rmse=float(np.sqrt(np.mean(v[mask]**2))))
        env.close()
        for seed in range(20):
            rows.extend(compare(name, seed, v, mask))
    (out/'references.json').write_text(json.dumps(refs, indent=2))
    (out/'runs.json').write_text(json.dumps(rows, indent=2))
    summary = []
    for name in refs:
        for budget in BUDGETS:
            for method in ('td0', 'td08', 'mc'):
                group = [r for r in rows if r['environment']==name and
                         r['transitions']==budget and r['method']==method]
                errors = np.array([r['rmse'] for r in group])
                summary.append(dict(environment=name, transitions=budget, method=method,
                                    rmse_mean=float(errors.mean()),
                                    rmse_sd=float(errors.std(ddof=1)),
                                    completed_episodes_range=[min(r['completed_episodes'] for r in group),
                                                              max(r['completed_episodes'] for r in group)]))
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'references': refs, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
