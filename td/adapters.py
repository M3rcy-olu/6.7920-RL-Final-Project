"""Environment adapters: how each environment's state becomes a feature vector phi(s).

An adapter owns three things the learner must not know about:

* how to build the env (including the step limit and any reward wrapper, D5),
* the feature encoder phi(obs, env) for the true Markov state (D11),
* the sizes ``n_features`` and ``n_actions``.

Only environments with a Discrete observation (FrozenLake, CliffWalking) have
an adapter so far: phi is the one-hot index of the observation. Minigrid and
Miniworld adapters are still to be designed (decision D11).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
from gymnasium import spaces


@dataclass
class Adapter:
    env_id: str
    make: Callable[[], gym.Env]
    n_features: int
    n_actions: int
    features: Callable[[object, gym.Env], object]


TOY_TEXT = ("FrozenLake-v1", "CliffWalking-v1")


def _discrete_obs_adapter(env_id: str, max_steps: int, **make_kwargs) -> Adapter:
    def make() -> gym.Env:
        return gym.make(env_id, max_episode_steps=max_steps, **make_kwargs)

    probe = make()
    try:
        if not isinstance(probe.observation_space, spaces.Discrete):
            raise TypeError(f"{env_id} does not have a Discrete observation space")
        n_features = int(probe.observation_space.n)
        n_actions = int(probe.action_space.n)
    finally:
        probe.close()
    return Adapter(env_id, make, n_features, n_actions, lambda obs, env: int(obs))


def make_adapter(env_id: str, *, max_steps: int, **make_kwargs) -> Adapter:
    """Return the adapter for ``env_id``, or raise NotImplementedError if none exists yet."""
    if env_id in TOY_TEXT:
        return _discrete_obs_adapter(env_id, max_steps, **make_kwargs)
    raise NotImplementedError(
        f"No adapter for {env_id!r} yet. Minigrid/Miniworld adapters are pending decision D11."
    )
