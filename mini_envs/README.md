# mini_envs: Minigrid & Miniworld setup

Custom environments, a single `make_env` factory and helper scripts. They're built on
[Minigrid](https://github.com/Farama-Foundation/Minigrid) (2D grid) and
[Miniworld](https://github.com/Farama-Foundation/Miniworld) (3D rooms). Background:
Chevalier-Boisvert et al., *Minigrid & Miniworld* (arXiv:2306.13831).

## Install

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-mini.txt   # Windows
# .venv/bin/python -m pip install -r requirements-mini.txt     # Linux/macOS
.venv/Scripts/python -m pytest -q tests
```

Tested on Python 3.11.4, Windows 11, CPU only. Miniworld renders with OpenGL through pyglet.
On a headless Linux box, run under `xvfb-run -a` (see the Miniworld troubleshooting docs).

## Environments

| id | library | task | actions |
|---|---|---|---|
| `MiniGrid-Proj-KeyDoor-v0` | Minigrid | Fixed 8x8 two rooms: get the key, unlock the door, reach the goal | 7 |
| `MiniGrid-Proj-KeyDoor-Random-v0` | Minigrid | Same, but the door row, the key and the start are random each reset | 7 |
| `MiniGrid-Proj-GoToObj-v0` | Minigrid | "go to the {color} {type}": face the target from an adjacent cell | 3 |
| `MiniWorld-Proj-GoToObj-v0` | Miniworld | Same missions in 3D: get `near()` the target | 3 |
| `MiniWorld-Proj-TwoRoom-v0` | Miniworld | Two rooms joined by an opening; reach the red box | 3 |

Both GoToObj envs use the same colors, object types and mission strings (`mini_envs/common.py`).
Both use the same 3 actions: turn left, turn right, forward. That makes them a ready-made pair for
Minigrid→Miniworld transfer, as in section 3.1 of the paper.

All rewards are sparse and only given on success. Minigrid pays `1 - 0.9 * t / max_steps`.
Miniworld pays `1 - 0.2 * t / max_steps`. Hitting the step limit returns
`truncated=True, terminated=False`.

Every upstream `MiniGrid-*`, `BabyAI-*` and `MiniWorld-*` id also works (`python scripts/list_envs.py`).

## Using them

```python
from mini_envs import make_env

env = make_env("MiniGrid-Proj-GoToObj-v0", obs="image", max_steps=200, size=10, num_objs=4)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`make_env(env_id, obs=None, max_steps=None, render_mode=None, **env_kwargs)`:

- `obs`: the observation format.
  - Minigrid: `dict` (default), `image` (7x7x3 symbolic), `full` (whole grid, symbolic), `rgb`
    (56x56 pixels), `rgb_full`.
  - Miniworld: `image` (default, 80x60 RGB) or `dict` (`{"image", "mission"}`, the same keys as Minigrid).
- `max_steps`: the episode limit for either library. Minigrid calls this `max_steps` and Miniworld
  calls it `max_episode_steps`; the factory translates.
- `**env_kwargs` go to the constructor, e.g. `size`, `num_objs`, `randomize`, `obs_width`,
  `view="top"`.

## Scripts

```sh
python scripts/list_envs.py --custom
python scripts/run_env.py --env MiniWorld-Proj-GoToObj-v0 --episodes 3 --seed 0
python scripts/run_env.py --env MiniGrid-Proj-KeyDoor-v0 --kw size=10 randomize=True --save-frames outputs/keydoor
python scripts/manual_control.py --env MiniGrid-Proj-KeyDoor-v0          # play it yourself
python scripts/manual_control.py --env MiniWorld-Proj-TwoRoom-v0 --top-view
```

`run_env.py` runs a seeded uniform-random policy and prints JSON. For each episode it reports the
return, the step count, the terminated/truncated flags and the mission. `--save-frames` writes PNGs
and `episode.gif` for the first episode. `outputs/` is git-ignored.

## Creating a new environment

**Minigrid** (in `minigrid_envs.py`):

```python
class MyGridEnv(MiniGridEnv):
    def __init__(self, size=8, max_steps=None, **kwargs):
        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission),
            grid_size=size,
            max_steps=4 * size**2 if max_steps is None else max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)        # outer walls
        self.grid.vert_wall(width // 2, 0)              # inner wall (or grid.set(x, y, Wall()))
        self.put_obj(Goal(), width - 2, height - 2)     # fixed position
        self.place_obj(Key("yellow"))                   # random free cell
        self.place_agent()                              # random pose (or set agent_pos / agent_dir)
        self.mission = self._gen_mission()
```

- Objects are in `minigrid.core.world_object`: `Wall, Door, Key, Ball, Box, Goal, Lava`.
- Use `self._rand_int`, `self._rand_elem` and the like for randomness, so `reset(seed=...)` stays
  reproducible.
- To change reward or termination, override `step`. To change only the success reward, override
  `_reward`.
- To restrict actions, set `self.action_space = spaces.Discrete(n)` after `super().__init__`.

**Miniworld** (in `miniworld_envs.py`):

```python
class MyWorldEnv(MiniWorldEnv, utils.EzPickle):
    def __init__(self, size=8, max_episode_steps=250, **kwargs):
        self.size = size
        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(self, size=size, max_episode_steps=max_episode_steps, **kwargs)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)  # left/right/forward

    def _gen_world(self):
        room = self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)  # X-Z plane
        self.box = self.place_entity(Box(color="red"))   # random pose; pos=np.array([x, 0, z]) to fix
        self.place_agent()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.near(self.box):
            reward += self._reward()
            terminated = True
        return obs, reward, terminated, truncated, info
```

- Entities are in `miniworld.entity`: `Box, Ball, Key, ImageFrame, TextFrame`.
- Join rooms with `self.connect_rooms(a, b, min_z=..., max_z=..., max_y=2.2)`; see
  `TwoRoomWorldEnv`.
- Use `self.np_random` for randomness.

**Then register it** by adding one line to `CUSTOM_ENVS` in `mini_envs/__init__.py`:

```python
"MiniGrid-Proj-MyEnv-v0": ("mini_envs.minigrid_envs:MyGridEnv", {"size": 10}),
```

Check it with `python scripts/run_env.py --env MiniGrid-Proj-MyEnv-v0 --save-frames outputs/myenv`,
play it with `scripts/manual_control.py`, and add a test in `tests/test_mini_envs.py`. The
seeded-rollout and obs-mode tests cover every id in `CUSTOM_ENVS` automatically.

## Not done yet

- There's no tabular state wrapper. The TD(lambda) code on `jenny/rl-bootstrap` expects integer
  states, while Minigrid observations are image dicts.
- There are no learning agents (PPO, etc.), no transfer experiments and no reward-delay/noise
  wrappers.
