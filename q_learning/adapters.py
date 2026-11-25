"""Dispatch environment-specific encoding outside the learning algorithm."""

from gym_envs import adapters as gym_adapter

ENVS = (*gym_adapter.ENVS, "MiniGrid-Proj-KeyDoor-v0")


def make_env(env_id, max_steps):
    if env_id in gym_adapter.ENVS:
        return gym_adapter.make_env(env_id, max_steps)
    if env_id == ENVS[-1]:
        from mini_envs import adapters

        return adapters.make_env(max_steps)
    raise ValueError(f"Unsupported baseline: {env_id}")


def state(obs, env):
    if env.spec.id in gym_adapter.ENVS:
        return gym_adapter.state(obs, env)
    from mini_envs import adapters

    return adapters.state(obs, env)
