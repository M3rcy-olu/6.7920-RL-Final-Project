"""Full symbolic state for the fixed KeyDoor baseline."""

import gymnasium as gym

ENV_ID = "MiniGrid-Proj-KeyDoor-v0"


def make_env(max_steps):
    return gym.make(ENV_ID, max_steps=max_steps)


def state(obs, env):
    base = env.unwrapped
    key = (-1, -1)
    door = None
    for x in range(base.width):
        for y in range(base.height):
            obj = base.grid.get(x, y)
            if obj is not None and obj.type == "key":
                key = (x, y)
            if obj is not None and obj.type == "door":
                door = (int(obj.is_open), int(obj.is_locked))
    carrying = int(base.carrying is not None)
    # Time matters: MiniGrid's terminal reward depends on step_count.
    return (
        *map(int, base.agent_pos),
        int(base.agent_dir),
        *key,
        carrying,
        *door,
        int(base.step_count),
    )


# ---- chain-of-rooms worlds (D13) ------------------------------------------------
CHAIN_PREFIX = "MiniGrid-Proj-Chain-"
CHAIN_STATE_WIDTH = 6
CHAIN_ACTIONS = 5  # left, right, forward, pickup, toggle


def is_chain(env_id):
    return env_id.startswith(CHAIN_PREFIX)


def make_chain_env(env_id, max_steps, **kwargs):
    """A chain world that skips building image observations.

    Q-learning reads the Markov state from the simulator, never the image, so
    the env returns ``obs=None``; this makes stepping several times faster.
    Extra kwargs (e.g. ``feedback_every``) go to the world's constructor.
    """
    env = gym.make(env_id, max_steps=max_steps, disable_env_checker=True, **kwargs)
    env.unwrapped._skip_obs = True
    return env


def _encode_chain(full):
    """Markov state -> 6 ints, dropping the loss field.

    Layout: x, y, direction, carried key colour index (-1 = none), frontier
    room, opened door of that room (-1 = none). With feedback after every room
    the loss is fixed by (frontier, opened door), so nothing is lost for
    training; leaving it out also lets one frozen policy act under any feedback
    setting.
    """
    from minigrid.core.constants import COLOR_NAMES

    x, y, d, carried, frontier, opened, _loss = full
    return (
        int(x),
        int(y),
        int(d),
        -1 if carried is None else COLOR_NAMES.index(carried),
        int(frontier),
        -1 if opened is None else int(opened),
    )


def chain_state(env):
    return _encode_chain(env.unwrapped.markov_state())


def chain_live_states(env):
    """Encoded states of every reachable non-terminal state."""
    from mini_envs.chain_rooms import reachable_states

    base = env.unwrapped
    skip = getattr(base, "_skip_obs", False)
    try:
        states = reachable_states(base)
    finally:
        base._skip_obs = skip
    return {_encode_chain(s) for s in states if s[4] != base.n_rooms}
