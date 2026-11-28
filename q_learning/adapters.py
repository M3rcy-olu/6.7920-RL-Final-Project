"""Dispatch environment-specific encoding outside the learning algorithm."""

from gym_envs import adapters as gym_adapter
from mini_envs import adapters as mini_adapter
from mini_envs.chain_worlds import CHAIN_WORLDS, COST_WORLDS

KEY_DOOR = "MiniGrid-Proj-KeyDoor-v0"
CHAIN_ENVS = (*CHAIN_WORLDS, *COST_WORLDS)
ENVS = (*gym_adapter.ENVS, KEY_DOOR, *CHAIN_ENVS)


def spec(env_id):
    """(state width, number of actions) of the saved policy format."""
    if env_id in gym_adapter.ENVS:
        return 1, 4
    if env_id == KEY_DOOR:
        return 9, 7
    if env_id in CHAIN_ENVS:
        return mini_adapter.CHAIN_STATE_WIDTH, mini_adapter.CHAIN_ACTIONS
    raise ValueError(f"Unsupported baseline: {env_id}")


def make_env(env_id, max_steps, **kwargs):
    if env_id in gym_adapter.ENVS:
        return gym_adapter.make_env(env_id, max_steps)
    if env_id == KEY_DOOR:
        return mini_adapter.make_env(max_steps)
    if env_id in CHAIN_ENVS:
        return mini_adapter.make_chain_env(env_id, max_steps, **kwargs)
    raise ValueError(f"Unsupported baseline: {env_id}")


def state(obs, env):
    env_id = env.spec.id
    if env_id in gym_adapter.ENVS:
        return gym_adapter.state(obs, env)
    if env_id in CHAIN_ENVS:
        return mini_adapter.chain_state(env)
    return mini_adapter.state(obs, env)


def live_states(env_id, env):
    """Every non-terminal state, for envs where the policy must cover them all."""
    if env_id in CHAIN_ENVS and hasattr(env.unwrapped, "markov_state"):
        return mini_adapter.chain_live_states(env)
    return set()
