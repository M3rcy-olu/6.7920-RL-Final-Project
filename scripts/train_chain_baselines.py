"""Train the study baselines using one budget and retry rule."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_envs.chain_rooms import plan
from mini_envs.chain_worlds import CHAIN_WORLDS, COST_WORLDS
from q_learning import Policy, evaluate, train
from q_learning.adapters import make_env

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'q_learning' / 'policies'
RESULTS = OUT / 'chain_results.json'


def main():
    results = []
    worlds = {**CHAIN_WORLDS, **COST_WORLDS}
    for env_id, spec in worlds.items():
        if spec['family'] == 'matter':
            continue
        env = make_env(env_id, 100000)
        optimal = len(plan(env.unwrapped))
        env.close()
        limit = 50 * optimal
        name = env_id.removeprefix('MiniGrid-Proj-Chain-').removesuffix('-v0')
        path = OUT / ('chain_' + name.lower().replace('-', '_') + '.json')
        attempts = []
        for episodes in (2000, 4000, 8000, 16000):
            start = time.perf_counter()
            policy = train(env_id, episodes=episodes, max_steps=limit,
                           seed=0, alpha=0.2, gamma=0.99, initial_value=0)
            seconds = time.perf_counter() - start
            metrics = evaluate(policy, episodes=100, seed=100000)
            passed = (metrics['success_rate'] == 1.0
                      and metrics['unseen_episodes'] == 0
                      and metrics['mean_steps'] <= 1.1 * optimal)
            attempts.append(dict(
                training_episodes=episodes, training_seconds=seconds,
                passed=passed, **metrics))
            print(name, episodes, round(seconds, 2), metrics,
                  'PASS' if passed else 'RETRY', flush=True)
            if passed:
                policy.save(path)
                assert Policy.load(path) == policy
                assert evaluate(Policy.load(path)) == metrics
                break
        results.append(dict(world=name, env_id=env_id, optimal_steps=optimal,
                            max_steps=limit, file=str(path.relative_to(ROOT)),
                            passed=passed, attempts=attempts))
        RESULTS.write_text(json.dumps(results, indent=2) + '\n')
    assert len(results) == 14
    print(f"{sum(row['passed'] for row in results)}/14 worlds passed")


if __name__ == '__main__':
    main()
