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
