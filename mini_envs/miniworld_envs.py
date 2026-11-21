"""Custom Miniworld environments.

Each class follows the pattern from the Miniworld tutorial
(https://miniworld.farama.org/main/content/create_env/):

1. Subclass ``MiniWorldEnv`` (and ``gymnasium.utils.EzPickle`` so vectorised
   envs can pickle it).
2. Override ``_gen_world()`` to add rooms (``add_rect_room``,
   ``connect_rooms``), entities (``place_entity``) and the agent
   (``place_agent``). Movement is in the X-Z plane; Y is up.
3. Override ``step`` to add task reward / termination, usually with
   ``self.near(entity)``.

Observations are RGB images of shape ``(obs_height, obs_width, 3)``
(80x60 by default). The registered ids are in ``mini_envs/__init__.py``.
"""
from __future__ import annotations

from gymnasium import spaces, utils
from miniworld.entity import Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv

from mini_envs.common import COLORS, OBJ_TYPES, goto_mission

_ENT_CLASSES = {"ball": Ball, "box": Box, "key": Key}


class TwoRoomWorldEnv(MiniWorldEnv, utils.EzPickle):
    """Two rooms joined by an opening. Reach the red box in the far room.

    3D counterpart of ``KeyDoorEnv``, without the key, since Miniworld has no
    lockable doors.

    Args:
        size: Side length of each square room, in world units.
        max_episode_steps: Episode limit; reaching it gives ``truncated=True``.

    Actions: turn left / turn right / move forward (``Discrete(3)``).
    Reward: Miniworld default ``1 - 0.2 * step_count / max_episode_steps`` on
    reaching the box, 0 otherwise.
    """

    def __init__(self, size: int = 6, max_episode_steps: int = 300, **kwargs):
        if size < 4:
            raise ValueError("TwoRoomWorldEnv needs size >= 4")
        self.size = size
        self.mission = goto_mission("red", "box")
        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(
            self, size=size, max_episode_steps=max_episode_steps, **kwargs
        )
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self) -> None:
        s = self.size
        left = self.add_rect_room(min_x=0, max_x=s, min_z=0, max_z=s)
        right = self.add_rect_room(min_x=s + 1, max_x=2 * s + 1, min_z=0, max_z=s)
        # 2-unit-wide opening in the middle of the shared wall.
        self.connect_rooms(left, right, min_z=s / 2 - 1, max_z=s / 2 + 1, max_y=2.2)

        self.box = self.place_entity(Box(color="red"), room=right)
        self.place_agent(room=left)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.near(self.box):
            reward += self._reward()
            terminated = True
        return obs, reward, terminated, truncated, info


class GoToObjectWorldEnv(MiniWorldEnv, utils.EzPickle):
    """One room with several objects. Go to the object named in the mission.

    Miniworld half of the paired GoToObj task from the Minigrid & Miniworld
    paper (section 3.1). It uses the same colors, object types and mission
    strings as ``GoToObjectGridEnv``.

    The mission is exposed as ``env.unwrapped.mission`` and ``info["mission"]``.
    To get Minigrid-style ``{"image", "mission"}`` dict observations, wrap it
    with ``mini_envs.wrappers.MiniWorldMissionWrapper`` (or use
    ``make_env(..., obs="dict")``).

    Args:
        size: Side length of the square room, in world units.
        num_objs: Number of distinct (color, type) objects in the room.
        max_episode_steps: Episode limit.

    Actions: turn left / turn right / move forward (``Discrete(3)``).
    Reward: Miniworld default on getting near the target, 0 otherwise.
    Touching a wrong object does nothing.
    """

    def __init__(
        self, size: int = 8, num_objs: int = 3, max_episode_steps: int = 250, **kwargs
    ):
        if not 1 <= num_objs <= len(COLORS) * len(OBJ_TYPES):
            raise ValueError("num_objs out of range")
        self.size = size
        self.num_objs = num_objs
        self.mission = ""
        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(
            self,
            size=size,
            num_objs=num_objs,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self) -> None:
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        rng = self.np_random
        combos = [(c, t) for c in COLORS for t in OBJ_TYPES]
        picks = rng.choice(len(combos), size=self.num_objs, replace=False)
        self.objects = []
        for i in picks:
            color, obj_type = combos[i]
            ent = self.place_entity(_ENT_CLASSES[obj_type](color=color))
            self.objects.append((color, obj_type, ent))

        self.target_color, self.target_type, self.target = self.objects[
            int(rng.integers(len(self.objects)))
        ]
        self.mission = goto_mission(self.target_color, self.target_type)
        self.place_agent()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["mission"] = self.mission
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.near(self.target):
            reward += self._reward()
            terminated = True
        info["mission"] = self.mission
        return obs, reward, terminated, truncated, info
