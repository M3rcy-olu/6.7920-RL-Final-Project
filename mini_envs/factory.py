"""One entry point for building Minigrid and Miniworld environments.

    from mini_envs import make_env
    env = make_env("MiniGrid-Proj-GoToObj-v0", obs="image")
    obs, info = env.reset(seed=0)

``make_env`` accepts every registered MiniGrid-* / MiniWorld-* id, the
built-in ones as well as ours. It also hides the differences between the two
libraries:

* ``max_steps`` sets the episode limit. Minigrid calls it ``max_steps`` and
  Miniworld calls it ``max_episode_steps``.
* ``obs`` chooses the observation format (see ``OBS_MODES``).
"""
from __future__ import annotations

import gymnasium as gym
from minigrid.wrappers import (
    FullyObsWrapper,
    ImgObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
)

import mini_envs  # noqa: F401  (registers our custom env ids)
from mini_envs.wrappers import MiniWorldMissionWrapper

OBS_MODES = {
    "minigrid": {
        "dict": "default {'image', 'direction', 'mission'}; image is the 7x7x3 symbolic partial view",
        "image": "partial symbolic view only, (7, 7, 3) uint8",
        "full": "full-grid symbolic view, (W, H, 3) uint8",
        "rgb": "partial view rendered as pixels, (56, 56, 3) uint8",
        "rgb_full": "whole grid rendered as pixels, (8W, 8H, 3) uint8",
    },
    "miniworld": {
        "image": "default first-person RGB, (obs_height, obs_width, 3) uint8",
        "dict": "{'image', 'mission'}, same keys as Minigrid",
    },
}


def library_of(env_id: str) -> str:
    """Return 'minigrid' or 'miniworld' for a registered env id."""
    if env_id.startswith("MiniGrid-") or env_id.startswith("BabyAI-"):
        return "minigrid"
    if env_id.startswith("MiniWorld-"):
        return "miniworld"
    raise ValueError(f"{env_id!r} is not a MiniGrid-/BabyAI-/MiniWorld- id")


def list_env_ids(library: str | None = None, custom_only: bool = False) -> list[str]:
    """Registered env ids, optionally filtered to one library or to our envs."""
    ids = []
    for env_id, spec in gym.registry.items():
        try:
            lib = library_of(env_id)
        except ValueError:
            continue
        if library and lib != library:
            continue
        if custom_only and not str(spec.entry_point).startswith("mini_envs."):
            continue
        ids.append(env_id)
    return sorted(ids)


def make_env(
    env_id: str,
    *,
    obs: str | None = None,
    max_steps: int | None = None,
    render_mode: str | None = None,
    **env_kwargs,
) -> gym.Env:
    """Build a Minigrid or Miniworld env with a chosen observation format.

    Args:
        env_id: Any registered id, e.g. ``"MiniGrid-Proj-KeyDoor-v0"``,
            ``"MiniWorld-OneRoom-v0"``.
        obs: Observation mode. Defaults to ``"dict"`` for Minigrid and
            ``"image"`` for Miniworld (each library's native format). See
            ``OBS_MODES``.
        max_steps: Episode step limit (``truncated=True`` when hit). None
            keeps the env's own default.
        render_mode: ``None``, ``"rgb_array"`` or ``"human"``.
        **env_kwargs: Passed to the env constructor (e.g. ``size=10``,
            ``num_objs=4``, ``randomize=True``, ``obs_width=128``).
    """
    lib = library_of(env_id)
    obs = obs or ("dict" if lib == "minigrid" else "image")
    if obs not in OBS_MODES[lib]:
        raise ValueError(f"obs={obs!r} not valid for {lib}; choose from {list(OBS_MODES[lib])}")

    if max_steps is not None:
        env_kwargs["max_steps" if lib == "minigrid" else "max_episode_steps"] = max_steps

    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)

    if lib == "minigrid":
        if obs == "image":
            env = ImgObsWrapper(env)
        elif obs == "full":
            env = ImgObsWrapper(FullyObsWrapper(env))
        elif obs == "rgb":
            env = ImgObsWrapper(RGBImgPartialObsWrapper(env))
        elif obs == "rgb_full":
            env = ImgObsWrapper(RGBImgObsWrapper(env))
    elif obs == "dict":
        env = MiniWorldMissionWrapper(env)
    return env
