"""Full symbolic states for project MiniGrid tasks."""

import gymnasium as gym
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

ENV_ID = "MiniGrid-Proj-KeyDoor-v0"
GOTO_ID = "MiniGrid-Proj-GoToObj-v0"


def make_env(max_steps, env_id=ENV_ID):
    if env_id not in (ENV_ID, GOTO_ID):
        raise ValueError(f"Unsupported MiniGrid baseline: {env_id}")
    return gym.make(env_id, max_steps=max_steps)


def state(obs, env):
    if env.spec.id == GOTO_ID:
        return goto_state(env)
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


def goto_state(env):
    """Encode the full grid and target, not a partial camera observation."""
    base = env.unwrapped
    if base.action_space.n != 3:
        raise ValueError("GoToObject encoder requires movement-only actions")
    return (
        int(base.width),
        int(base.height),
        *map(int, base.agent_pos),
        int(base.agent_dir),
        int(base.step_count),
        *map(int, base.target_pos),
        int(COLOR_TO_IDX[base.target_color]),
        int(OBJECT_TO_IDX[base.target_type]),
        *map(int, base.grid.encode().ravel()),
    )
