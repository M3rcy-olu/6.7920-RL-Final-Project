"""Small fixed-policy evaluation baseline, not a control algorithm."""
import argparse
import json
from environments import make_env
from td_lambda import td_update
import numpy as np


def evaluate(name, *, episodes=10, seed=0, alpha=0.1, gamma=0.99, lam=0.8):
    if episodes < 1 or not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('Invalid evaluation parameters')
    env = make_env(name)
    rng = np.random.default_rng(seed)
    values = np.zeros(env.observation_space.n)
    results = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            traces = np.zeros_like(values)
            total, steps = 0.0, 0
            while True:
                action = int(rng.integers(env.action_space.n))
                nxt, reward, terminated, truncated, _ = env.step(action)
                td_update(values, traces, state, reward, nxt, terminated,
                          alpha=alpha, gamma=gamma, lam=lam)
                total += reward
                steps += 1
                state = nxt
                if terminated or truncated:
                    results.append(dict(episode=episode, return_=total, steps=steps,
                                        terminated=bool(terminated), truncated=bool(truncated)))
                    break
    finally:
        env.close()
    return dict(environment=name, policy='uniform_random', seed=seed,
                alpha=alpha, gamma=gamma, lam=lam, episodes=results,
                values=values.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=['FrozenLake-v1', 'CliffWalking-v1'], default='FrozenLake-v1')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.env, episodes=args.episodes, seed=args.seed), indent=2))
