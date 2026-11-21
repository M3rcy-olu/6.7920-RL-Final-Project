"""Project environments built on Minigrid (2D) and Miniworld (3D).

Importing this package registers the upstream Minigrid / Miniworld ids and
our custom ids below with Gymnasium, so ``gym.make(<id>)`` works.

To add an env: write the class in ``minigrid_envs.py`` or
``miniworld_envs.py``, then add a ``register(...)`` line here. See
``mini_envs/README.md``.
"""
from __future__ import annotations

import minigrid  # noqa: F401  (registers MiniGrid-* / BabyAI-* ids)
import miniworld  # noqa: F401  (registers MiniWorld-* ids)
from gymnasium.envs.registration import register, registry

CUSTOM_ENVS = {
    # id: (entry point, default kwargs)
    "MiniGrid-Proj-KeyDoor-v0": ("mini_envs.minigrid_envs:KeyDoorEnv", {}),
    "MiniGrid-Proj-KeyDoor-Random-v0": (
        "mini_envs.minigrid_envs:KeyDoorEnv",
        {"randomize": True},
    ),
    "MiniGrid-Proj-GoToObj-v0": ("mini_envs.minigrid_envs:GoToObjectGridEnv", {}),
    "MiniWorld-Proj-TwoRoom-v0": ("mini_envs.miniworld_envs:TwoRoomWorldEnv", {}),
    "MiniWorld-Proj-GoToObj-v0": ("mini_envs.miniworld_envs:GoToObjectWorldEnv", {}),
}

for _env_id, (_entry_point, _kwargs) in CUSTOM_ENVS.items():
    if _env_id not in registry:
        register(id=_env_id, entry_point=_entry_point, kwargs=_kwargs)

from mini_envs.factory import OBS_MODES, library_of, list_env_ids, make_env  # noqa: E402

__all__ = ["CUSTOM_ENVS", "OBS_MODES", "library_of", "list_env_ids", "make_env"]
