"""Integer-state adapters for Gymnasium's toy-text games."""

import gymnasium as gym

ENVS = ("FrozenLake-v1", "CliffWalking-v1")


def make_env(env_id, max_steps):
    if env_id not in ENVS:
        raise ValueError(f"Unsupported Gym baseline: {env_id}")
    return gym.make(env_id, max_episode_steps=max_steps)


def state(obs, env):
    return (int(obs),)
