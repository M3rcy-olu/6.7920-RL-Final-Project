# Tabular Q-learning baselines

Supported: FrozenLake-v1 (default slippery 4x4), CliffWalking-v1 and fixed
MiniGrid-Proj-KeyDoor-v0. No pixels, neural networks or parameterized models.
GoToObject has encoder and policy support, but no validated learned policy.
Random KeyDoor and MiniWorld are intentionally excluded.

## Organization

- `gym_envs/adapters.py`: Gym construction and integer state encoding.
- `mini_envs/adapters.py`: MiniGrid construction and full symbolic state encoding.
- `td/`: existing value estimation, unchanged.
- `q_learning/`: shared training, policy serialization and frozen evaluation.
- `q_learning/policies/`: learned action tables in one versioned JSON format.

The existing TD runner still defaults to random actions. Loading these files does
not automatically change it. TD's MiniGrid feature adapter, exact DP comparisons
and perturbation experiments remain separate work.

## Reproduce

Install `requirements-mini.txt`, plus pytest for tests. The existing mini_envs
package imports MiniWorld even for MiniGrid; Debian also needs libglu1-mesa.
No renderer or GPU is needed. Tested dependency versions and results are in
`policies/README.md`.

```sh
python -m q_learning --env CliffWalking-v1 --episodes 10000 --max-steps 256 --seed 0 --output q_learning/policies/cliff_walking.json
python -m q_learning --env FrozenLake-v1 --episodes 40000 --max-steps 256 --seed 0 --output q_learning/policies/frozen_lake.json
python -m q_learning --env MiniGrid-Proj-KeyDoor-v0 --episodes 30000 --max-steps 256 --seed 0 --alpha 1 --initial-value 0.1 --progress --output q_learning/policies/key_door.json
python -m q_learning --load q_learning/policies/key_door.json
python -m pytest -q
```

Standard online one-step off-policy tabular Q-learning uses alpha=0.2, gamma=0.99
and epsilon linearly decreasing from 1 to 0.05 over the first 80% of episodes. Exploration
breaks ties randomly; exported greedy actions break ties by lowest action index.
Initial values default to zero; the KeyDoor command uses 0.1 and alpha=1.
Training resets use seed + episode. Frozen evaluation defaults to 100 episodes
starting at seed 100000, with no exploration or learning. Reported returns are
undiscounted rewards, not the discounted training objective.

Gym truncations are rollout cutoffs and bootstrap. KeyDoor's horizon is part of
the task: truncation does not bootstrap, and time is in state because its terminal
reward depends on elapsed steps. Results are empirical, not optimality proofs.

## Use in a game rollout

```python
from q_learning import Policy
from q_learning.adapters import make_env

policy = Policy.load("q_learning/policies/key_door.json")
env = make_env(policy.env_id, policy.max_steps)
try:
    obs, info = env.reset(seed=100000)
    while True:
        action = policy.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

One schema for all games: version, env_id, max_steps and sorted actions pairs
`[[state_as_list, discrete_action], ...]`. Only greedy actions are stored, not Q
values or exploration state. This is a rollout artifact, not a training checkpoint.
Gym state is a one-element integer tuple. KeyDoor state is x, y, direction, key x,
key y, carrying flag, door-open flag, door-locked flag and step count. A carried
key has position (-1, -1).

The policy rejects the wrong environment, mismatched KeyDoor horizon and invalid
actions. Unvisited states raise KeyError rather than silently selecting random
actions. Evaluation reports these as unseen_episodes and ends those rollouts.
Perturbations must keep actions frozen and report coverage failures explicitly.
Matching IDs do not certify matching layouts or transitions: these tables cover
only the documented baseline settings, not arbitrary maps or unseen perturbations.

## GoToObject encoder

`MiniGrid-Proj-GoToObj-v0` supports the default movement-only actions (left,
right, forward). Its integer tuple is width, height, agent x/y, direction,
step count, target x/y, target color/type indices, then the complete grid's
object/color/state triples flattened in NumPy C order from `grid.encode()`.
The header has 10 integers; the grid adds `3 * width * height` integers.
This is privileged full symbolic state, not the agent's partial image.
Target fields distinguish missions on the same layout. Walls and all objects
are included so randomized layouts do not alias. Time is included because
terminal rewards depend on elapsed steps. Non-movement action sets are rejected.

The shared CLI can train this task with `--env MiniGrid-Proj-GoToObj-v0`,
but this PR supplies no trained GoToObject artifact or success claim. With
randomized layouts and targets, a full-state table may have poor coverage.
An encoder passing correctness tests does not establish learnability at a
practical episode budget. Exact DP and TD feature integration remain separate.
