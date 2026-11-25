# Tabular Q-learning baselines

Supported: FrozenLake-v1 (default slippery 4x4), CliffWalking-v1, fixed
MiniGrid-Proj-KeyDoor-v0 and every chain-of-rooms world
(`MiniGrid-Proj-Chain-*-v0`, see below). No pixels, neural networks or parameterized models.
Random KeyDoor, GoToObj and MiniWorld are intentionally excluded.

## Organization

- `gym_envs/adapters.py`: Gym construction and integer state encoding.
- `mini_envs/adapters.py`: fixed KeyDoor construction and symbolic state encoding.
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

## Chain-of-rooms worlds

All 18 `MiniGrid-Proj-Chain-*-v0` ids (`mini_envs/chain_worlds.py`; list them
with `python scripts/list_envs.py`) train with the same command, for example:

```sh
python -m q_learning --env MiniGrid-Proj-Chain-Base-v0 --episodes <N> --max-steps <limit> --seed 0 --output q_learning/policies/chain_base.json
```

- **Standard version:** the registered world as is, which pays reward after
  every room (`feedback_every=1`). Train on that.
- **State:** 6 ints read from the simulator: x, y, direction, carried key colour
  index (-1 = none), frontier room, opened door of that room (-1 = none). The
  world's "loss in the current gap" field is left out: with feedback after every
  room it is fixed by (frontier, opened door), so training loses nothing (tested),
  and the frozen policy can act unchanged in the same world under any feedback
  setting, e.g. `gym.make(id, feedback_every=None)`.
- **Step limit:** a cutoff bootstraps (D4), unlike KeyDoor, because time is not
  part of these tasks. Pick `--max-steps` from the step-limit sweep
  (`scripts/chain_step_limits.py`, results in the decision doc).
- **Coverage:** after training, every reachable non-terminal state that was never
  visited gets the tie-break action (lowest index), so the policy file covers
  the whole state space and an epsilon-greedy version or exact DP is defined
  everywhere. Rollouts never raise `KeyError` in these worlds.
- **Success** in `evaluate` means reaching the end with the full return of 1.
- The env skips building image observations (`obs` is `None`), because the
  policy reads the Markov state; this makes training several times faster.
