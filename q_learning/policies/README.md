# Saved policy provenance

Empirical tabular baselines, not optimality certificates. Evaluation uses frozen
greedy actions for 100 episodes, with reset seeds 100000 through 100099.
Training uses reset seed 0 + episode and gamma 0.99. Alpha is 0.2 for Gym
and 1 for KeyDoor. Epsilon decreases
linearly from 1 to 0.05 over the first 80% of episodes. All horizons are 256 steps.

| Environment | Episodes | Initial Q | Success rate | Mean return | Mean steps | Unseen episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CliffWalking-v1 | 10000 | 0 | 1.00 | -13.00 | 13.00 | 0 |
| FrozenLake-v1, slippery 4x4 | 40000 | 0 | 0.77 | 0.77 | 47.14 | 0 |

KeyDoor's 30000-episode run with alpha=1 and initial Q=0.1 achieved 100/100
frozen evaluation successes, mean return 0.94375 and 16 steps, with no unseen
episodes. It collected 4291 goal rewards during exploration. Earlier runs with
alpha=0.2 failed frozen evaluation, including 10000 episodes with initial Q=0.1.
Training uses standard online one-step Q-learning, without replay,
demonstrations or reward shaping.

KeyDoor reset seeds do not generate held-out layouts: this task has a fixed
layout. Reward depends on elapsed steps, so state includes time and truncation
is terminal. Gym truncations bootstrap. See the parent README for the schema
and coverage limitations.

Tested with Python 3.13.14, gymnasium 1.4.0, minigrid 3.2.0, numpy 2.3.5,
miniworld 2.1.0, pyglet 1.5.31 and pytest 9.0.3. The existing mini_envs package
imports MiniWorld eagerly; Debian needs libglu1-mesa for headless MiniGrid runs.
