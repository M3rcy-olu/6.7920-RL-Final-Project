"""Vocabulary shared by the Minigrid and Miniworld environments.

Keeping one list of colors and object types means the paired environments
produce the same mission strings ("go to the red ball") in both libraries.
That matters for transfer experiments, where a mission encoder trained on
Minigrid should see the same vocabulary on Miniworld.
"""
from __future__ import annotations

# Colors that exist in both minigrid.core.constants.COLOR_NAMES and
# miniworld.entity.COLOR_NAMES.
COLORS: tuple[str, ...] = ("red", "green", "blue", "purple", "yellow", "grey")

# Object types that exist in both libraries.
OBJ_TYPES: tuple[str, ...] = ("ball", "box", "key")


def goto_mission(color: str, obj_type: str) -> str:
    return f"go to the {color} {obj_type}"
