"""Observation wrappers that give Minigrid and Miniworld matching interfaces."""
from __future__ import annotations

import string

import gymnasium as gym
from gymnasium import spaces


class MiniWorldMissionWrapper(gym.ObservationWrapper):
    """Turn a Miniworld RGB observation into ``{"image": ..., "mission": str}``.

    This gives a Miniworld env the same dict keys as a Minigrid env, so a
    policy with an image encoder plus a mission encoder can run on either.
    The paper's transfer case study needed the same change (section 3.3,
    item 2). The mission is read from ``env.unwrapped.mission``. Envs without
    one get an empty string.
    """

    def __init__(self, env: gym.Env, max_mission_len: int = 96):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "image": env.observation_space,
                "mission": spaces.Text(
                    max_length=max_mission_len,
                    min_length=0,  # envs without a mission give ""
                    charset=string.ascii_letters + string.digits + " ,.-",
                ),
            }
        )

    def observation(self, observation):
        return {
            "image": observation,
            "mission": getattr(self.env.unwrapped, "mission", ""),
        }
