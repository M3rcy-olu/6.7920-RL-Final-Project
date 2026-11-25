"""Custom Minigrid environments.

Each class follows the pattern from the Minigrid tutorial
(https://minigrid.farama.org/main/content/create_env_tutorial/):

1. Subclass ``MiniGridEnv``.
2. Build a ``MissionSpace`` from a static ``_gen_mission`` function.
3. Override ``_gen_grid(width, height)`` to lay out walls, objects, the goal
   and the agent.
4. Optionally override ``step`` to change termination or reward.

The registered ids are in ``mini_envs/__init__.py``.
"""
from __future__ import annotations

from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

from mini_envs.common import COLORS, OBJ_TYPES, goto_mission

_OBJ_CLASSES = {"ball": Ball, "box": Box, "key": Key}


class KeyDoorEnv(MiniGridEnv):
    """Two rooms split by a wall. Pick up the key, unlock the door, reach the goal.

    This is the tutorial ``SimpleEnv`` with its hard-coded numbers turned into
    arguments.

    Args:
        size: Side length of the square grid, walls included (min 6).
        randomize: If False, the layout is fixed (key, door and start never
            move between resets). If True, the door row, the key and the agent
            start are sampled in the left room each reset.
        agent_start_pos / agent_start_dir: Fixed start used when
            ``randomize`` is False.
        max_steps: Episode limit; reaching it gives ``truncated=True``.
            Defaults to ``4 * size**2``.

    Actions are the standard 7 Minigrid actions (pickup and toggle are needed).
    Reward is the Minigrid default ``1 - 0.9 * step_count / max_steps`` on
    reaching the goal, 0 otherwise.
    """

    def __init__(
        self,
        size: int = 8,
        randomize: bool = False,
        agent_start_pos: tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        **kwargs,
    ):
        if size < 6:
            raise ValueError("KeyDoorEnv needs size >= 6")
        self.randomize = randomize
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission),
            grid_size=size,
            see_through_walls=True,  # faster; walls still block movement
            max_steps=4 * size**2 if max_steps is None else max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Vertical wall splitting the grid into a left and right room.
        split_x = width // 2
        for y in range(height):
            self.grid.set(split_x, y, Wall())

        door_y = self._rand_int(1, height - 1) if self.randomize else height - 3
        self.grid.set(split_x, door_y, Door("yellow", is_locked=True))

        self.put_obj(Goal(), width - 2, height - 2)

        left_room = dict(top=(1, 1), size=(split_x - 1, height - 2))
        if self.randomize:
            self.place_obj(Key("yellow"), **left_room)
            self.place_agent(**left_room)
        else:
            self.put_obj(Key("yellow"), split_x - 2, height - 3)
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self.mission = self._gen_mission()


class GoToObjectGridEnv(MiniGridEnv):
    """One room with several objects. Go to the object named in the mission.

    Minigrid half of the paired GoToObj task from the Minigrid & Miniworld
    paper (section 3.1). The Miniworld half is
    ``mini_envs.miniworld_envs.GoToObjectWorldEnv``.

    Success is facing the target from an adjacent cell. That is the closest
    grid analogue of Miniworld's ``near()``. Facing a wrong object does
    nothing.

    Args:
        size: Side length of the square grid, walls included.
        num_objs: Number of distinct (color, type) objects in the room.
        movement_only: If True, only turn left / turn right / move forward are
            allowed (``Discrete(3)``). That matches the Miniworld action space.
        max_steps: Episode limit. Defaults to ``5 * size**2``.

    Observation: the standard Minigrid dict ``{"image", "direction", "mission"}``.
    """

    def __init__(
        self,
        size: int = 8,
        num_objs: int = 3,
        movement_only: bool = True,
        max_steps: int | None = None,
        **kwargs,
    ):
        if not 1 <= num_objs <= len(COLORS) * len(OBJ_TYPES):
            raise ValueError("num_objs out of range")
        self.num_objs = num_objs
        self.target_pos: tuple[int, int] | None = None
        self.target_color: str | None = None
        self.target_type: str | None = None
        super().__init__(
            mission_space=MissionSpace(
                mission_func=goto_mission,
                ordered_placeholders=[list(COLORS), list(OBJ_TYPES)],
            ),
            grid_size=size,
            see_through_walls=True,
            max_steps=5 * size**2 if max_steps is None else max_steps,
            **kwargs,
        )
        if movement_only:
            self.action_space = spaces.Discrete(self.actions.forward + 1)

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        placed: list[tuple[str, str, tuple[int, int]]] = []
        while len(placed) < self.num_objs:
            color = self._rand_elem(COLORS)
            obj_type = self._rand_elem(OBJ_TYPES)
            if any(c == color and t == obj_type for c, t, _ in placed):
                continue
            pos = self.place_obj(_OBJ_CLASSES[obj_type](color))
            placed.append((color, obj_type, tuple(pos)))

        self.place_agent()

        self.target_color, self.target_type, self.target_pos = placed[
            self._rand_int(0, len(placed))
        ]
        self.mission = goto_mission(self.target_color, self.target_type)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if tuple(self.front_pos) == self.target_pos:
            reward = self._reward()
            terminated = True
        return obs, reward, terminated, truncated, info
