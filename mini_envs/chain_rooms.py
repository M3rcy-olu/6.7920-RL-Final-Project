"""Chain-of-rooms maze for the three-factor TD(lambda) study (decision D13).

Layout (left to right)::

    [room 0] =corridors=> [room 1] =corridors=> ... [room n-1] =corridors=> [goal room]

* Every room has ``doors`` locked doors on its east wall, one key per door
  (matching colours) in its middle column, and an equal-length corridor from
  each door to the next room. So every choice leads forward, and the number
  of steps between rooms does not depend on which door is chosen.
* Room types: ``"c1"`` = one correct door (a choice that matters);
  ``"c2"`` = every door is correct (a choice that does not matter);
  a number ``c`` in (0, 1) = partial-cost room: one correct door, and a wrong
  door costs the fraction ``c`` of that gap's payout instead of all of it.
  ``"c1"`` is cost 1 and ``"c2"`` cost 0, so the rules are one rule.
* A key is used up when it opens its door. The room's other doors turn into
  walls and its leftover keys disappear, so the choice is final.
* Entering the next room seals the way back.
* Feedback: rooms are grouped into gaps of ``feedback_every`` rooms (``None``
  = one gap, feedback only at the end). Entering the room after a gap pays
  ``(1 / n_gaps) * max(0, 1 - loss)``, where ``loss`` sums the costs of wrong
  doors opened in that gap. A perfect episode therefore returns exactly 1
  (D5 with redistribution).
* Actions: left, right, forward, pickup, toggle (no drop).

The true Markov state (D11) is
``(x, y, direction, carried key colour, frontier room, opened door, loss)``.
``loss`` is the cost of wrong doors so far in the current gap; for worlds with
only c1/c2 rooms it is 0 or 1 (it used to be the boolean ``mistake``, and
False/True still compare equal to 0/1)
and can be read with ``markov_state()`` and restored with
``set_markov_state()``. ``plan`` and ``reachable_states`` search that state
space by breadth-first search.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

ROOM_TYPES = ("c1", "c2", "cp")  # "cp" = partial-cost room (given as a number)
MISSION = "pick a key, open a door, and reach the end"


class ChainRoomsEnv(MiniGridEnv):
    """Chain of rooms with key-locked exit doors and equal-length corridors.

    Args:
        rooms: Room types in order, each ``"c1"`` (one correct door),
            ``"c2"`` (all doors correct), or a number in (0, 1): a partial-cost
            room whose wrong door costs that fraction of the gap's payout.
        room_size: Interior side length of each square room (>= 2*doors - 1).
        doors: Exit doors (and keys) per room, b >= 2.
        corridor_len: Corridor cells between a door and the next room (>= 1).
        feedback_every: Rooms per feedback gap; ``None`` = feedback only at
            the end; 1 = feedback after every room.
        layout_seed: Fixes key colours, key order and which case-1 door is
            correct. The layout never changes between resets.
        max_steps: Step limit; ``None`` = a generous default until the D12
            step-limit test sets it per world.
    """

    def __init__(
        self,
        rooms: Sequence[str | float] = ("c1", "c2", "c1", "c2"),
        room_size: int = 5,
        doors: int = 2,
        corridor_len: int = 4,
        feedback_every: int | None = 1,
        layout_seed: int = 0,
        max_steps: int | None = None,
        **kwargs,
    ):
        rooms = tuple(rooms)
        costs = []
        for t in rooms:
            if t == "c1":
                costs.append(1.0)
            elif t == "c2":
                costs.append(0.0)
            elif isinstance(t, (int, float)) and not isinstance(t, bool) and 0 < t < 1:
                costs.append(float(t))
            else:
                raise ValueError('rooms must be a non-empty sequence of "c1", "c2" or a cost in (0, 1)')
        if not rooms:
            raise ValueError('rooms must be a non-empty sequence of "c1", "c2" or a cost in (0, 1)')
        if not 2 <= doors <= len(COLOR_NAMES):
            raise ValueError("doors must be between 2 and 6")
        if room_size < 2 * doors - 1:
            raise ValueError("room_size must be at least 2*doors - 1")
        if corridor_len < 1:
            raise ValueError("corridor_len must be >= 1")
        if feedback_every is not None and feedback_every < 1:
            raise ValueError("feedback_every must be >= 1 or None")

        self.room_types = tuple(t if isinstance(t, str) else "cp" for t in rooms)
        self.wrong_cost = costs
        self.n_rooms = len(rooms)
        self.room_size = room_size
        self.doors = doors
        self.corridor_len = corridor_len
        self.feedback_every = self.n_rooms if feedback_every is None else min(feedback_every, self.n_rooms)
        self.n_gaps = -(-self.n_rooms // self.feedback_every)

        rng = np.random.default_rng(layout_seed)
        s = room_size
        self.door_rows = [1 + round(i * (s - 1) / (doors - 1)) for i in range(doors)]
        self.door_colors = [list(rng.choice(COLOR_NAMES, size=doors, replace=False)) for _ in rooms]
        # key_rows[r][j] = row of the key that opens door j of room r
        self.key_rows = [list(rng.permutation(self.door_rows)) for _ in rooms]
        self.correct_door = [int(rng.integers(doors)) if c > 0 else None for c in costs]

        self._stride = s + 2 + corridor_len
        width = self.n_rooms * self._stride + 5  # + goal room (3 wide)
        height = s + 2
        if max_steps is None:
            max_steps = 10 * self.n_rooms * (2 * s + corridor_len + 8)
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: MISSION),
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs,
        )
        A = self.actions
        self._action_map = (A.left, A.right, A.forward, A.pickup, A.toggle)
        self.action_space = spaces.Discrete(len(self._action_map))

    # ---- geometry ---------------------------------------------------------
    def _west_wall_x(self, i: int) -> int:
        return i * self._stride

    def room_interior_x(self, i: int) -> tuple[int, int]:
        """Interior x range of room i (i == n_rooms is the goal room)."""
        wx = self._west_wall_x(i)
        return wx + 1, wx + (3 if i == self.n_rooms else self.room_size)

    def room_doors(self, i: int) -> list[tuple[int, int]]:
        ex = self._west_wall_x(i) + self.room_size + 1
        return [(ex, y) for y in self.door_rows]

    def room_keys(self, i: int) -> list[tuple[int, int]]:
        kx = self._west_wall_x(i) + 1 + self.room_size // 2
        return [(kx, y) for y in self.key_rows[i]]

    def room_entries(self, i: int) -> list[tuple[int, int]]:
        """Openings in the west wall of room i (i >= 1), one per corridor."""
        wx = self._west_wall_x(i)
        return [(wx, y) for y in self.door_rows]

    def _room_of_x(self, x: int) -> int | None:
        for i in range(self.n_rooms + 1):
            lo, hi = self.room_interior_x(i)
            if lo <= x <= hi:
                return i
        return None

    # ---- generation -------------------------------------------------------
    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        for x in range(width):
            for y in range(height):
                self.grid.set(x, y, Wall())
        n, s, L = self.n_rooms, self.room_size, self.corridor_len
        for i in range(n + 1):
            lo, hi = self.room_interior_x(i)
            for x in range(lo, hi + 1):
                for y in range(1, s + 1):
                    self.grid.set(x, y, Goal() if i == n else None)
            if i >= 1:
                for p in self.room_entries(i):
                    self.grid.set(*p, None)
        for i in range(n):
            for j, (dx, dy) in enumerate(self.room_doors(i)):
                color = self.door_colors[i][j]
                self.grid.set(dx, dy, Door(color, is_locked=True))
                for x in range(dx + 1, dx + 1 + L):
                    self.grid.set(x, dy, None)
            for j, (kx, ky) in enumerate(self.room_keys(i)):
                self.grid.set(kx, ky, Key(self.door_colors[i][j]))

        self.agent_pos = (self.room_interior_x(0)[0], (s + 1) // 2)
        self.agent_dir = 0
        self.mission = MISSION
        self.frontier, self.opened, self.mistake = 0, None, 0.0

    # ---- dynamics ----------------------------------------------------------
    def step(self, action):
        a = self._action_map[int(action)]
        door_idx = None
        if a == self.actions.toggle and self.frontier < self.n_rooms and self.opened is None:
            fwd = tuple(int(v) for v in self.front_pos)
            if fwd in self.room_doors(self.frontier):
                door_idx = self.room_doors(self.frontier).index(fwd)

        obs, _, terminated, truncated, info = super().step(a)
        reward = 0.0

        if door_idx is not None:
            door = self.grid.get(*self.room_doors(self.frontier)[door_idx])
            if isinstance(door, Door) and not door.is_locked:
                self._open(door_idx)

        room = self._room_of_x(int(self.agent_pos[0]))
        if room is not None and room == self.frontier + 1:
            reward = self._enter(room)
        terminated = terminated or self.frontier == self.n_rooms
        return obs, reward, terminated, truncated, info

    def _open(self, j: int) -> None:
        """Door j of the frontier room was unlocked with its key."""
        i = self.frontier
        self.carrying = None  # key used up
        for k, p in enumerate(self.room_doors(i)):
            self.grid.set(*p, None if k == j else Wall())
        for p in self.room_keys(i):
            if isinstance(self.grid.get(*p), Key):
                self.grid.set(*p, None)
        self.opened = j
        if self.correct_door[i] is not None and j != self.correct_door[i]:
            self.mistake = round(self.mistake + self.wrong_cost[i], 12)

    def _enter(self, i: int) -> float:
        """Agent first enters room i (i == n_rooms is the goal room)."""
        if i < self.n_rooms:
            for p in self.room_entries(i):
                self.grid.set(*p, Wall())
        reward = 0.0
        if i % self.feedback_every == 0 or i == self.n_rooms:
            reward = max(0.0, 1.0 - self.mistake) / self.n_gaps
            self.mistake = 0.0
        self.frontier, self.opened = i, None
        return reward

    # ---- Markov state ----------------------------------------------------
    def markov_state(self) -> tuple:
        carried = self.carrying.color if isinstance(self.carrying, Key) else None
        return (
            int(self.agent_pos[0]),
            int(self.agent_pos[1]),
            int(self.agent_dir),
            carried,
            self.frontier,
            self.opened,
            self.mistake,
        )

    def set_markov_state(self, state: tuple) -> None:
        """Rebuild the grid and agent for ``state``.

        Rooms behind the frontier are sealed and unreachable, so they are
        rebuilt as walls; dynamics match exactly, pixels there may differ.
        """
        x, y, d, carried, frontier, opened, mistake = state
        layout = (frontier, opened, carried)
        if layout != self._layout_key():
            # The grid differs only through frontier, opened door and which key
            # was picked up, so rebuild it only when one of those changes.
            self._gen_grid(self.width, self.height)
            for i in range(min(frontier, self.n_rooms)):
                for p in self.room_doors(i):
                    self.grid.set(*p, Wall())
                for p in self.room_keys(i):
                    self.grid.set(*p, None)
            for i in range(1, min(frontier, self.n_rooms) + 1):
                if i < self.n_rooms:
                    for p in self.room_entries(i):
                        self.grid.set(*p, Wall())
            self.frontier = frontier
            if opened is not None:
                self.opened = None
                self._open(opened)
            if carried is not None and frontier < self.n_rooms:
                j = self.door_colors[frontier].index(carried)
                self.grid.set(*self.room_keys(frontier)[j], None)  # it is in hand
            self.carrying = Key(carried) if carried is not None else None
            if self.carrying is not None:
                self.carrying.cur_pos = np.array([-1, -1])
        self.frontier, self.opened, self.mistake = frontier, opened, mistake
        self.agent_pos = (x, y)
        self.agent_dir = d
        self.step_count = 0

    def _layout_key(self) -> tuple:
        carried = self.carrying.color if isinstance(self.carrying, Key) else None
        return (getattr(self, "frontier", None), getattr(self, "opened", None), carried)

    def gen_obs(self):
        # Searches (plan, reachable_states) never read observations.
        if getattr(self, "_skip_obs", False):
            return None
        return super().gen_obs()


# ---- search helpers -----------------------------------------------------------
def _successors(env: ChainRoomsEnv, state: tuple):
    env._skip_obs = True
    try:
        for a in range(env.action_space.n):
            env.set_markov_state(state)
            _, r, terminated, _, _ = env.step(a)
            yield a, env.markov_state(), r, terminated
    finally:
        env._skip_obs = False


def reachable_states(env: ChainRoomsEnv) -> set[tuple]:
    """All Markov states reachable from the start (non-terminal and terminal)."""
    env.reset(seed=0)
    start = env.markov_state()
    seen, queue = {start}, deque([start])
    while queue:
        s = queue.popleft()
        for _, s2, _, terminated in _successors(env, s):
            if s2 not in seen:
                seen.add(s2)
                if not terminated:
                    queue.append(s2)
    env.reset(seed=0)
    return seen


def plan(env: ChainRoomsEnv, door_choices: dict[int, int] | None = None) -> list[int]:
    """Shortest action sequence to the end.

    With ``door_choices`` ({room: door}) those rooms must use the given doors
    and other rooms use correct doors; without it, every case-1 room uses its
    correct door, so the plan earns the full return of 1.
    """
    door_choices = door_choices or {}
    env.reset(seed=0)
    start = env.markov_state()
    parent = {start: None}
    queue = deque([start])
    goal = None
    while queue and goal is None:
        s = queue.popleft()
        for a, s2, _, terminated in _successors(env, s):
            if s2 in parent:
                continue
            frontier, opened = s2[4], s2[5]
            if opened is not None and frontier < env.n_rooms:
                want = door_choices.get(frontier, env.correct_door[frontier])
                if want is not None and opened != want:
                    continue
            parent[s2] = (s, a)
            if terminated:
                goal = s2
                break
            queue.append(s2)
    env.reset(seed=0)
    if goal is None:
        raise RuntimeError("no plan found")
    actions, s = [], goal
    while parent[s] is not None:
        s, a = parent[s]
        actions.append(a)
    return actions[::-1]


@dataclass(frozen=True)
class TransitionTable:
    """The world's exact dynamics over its reachable Markov states.

    ``next[s, a]``, ``reward[s, a]`` and ``terminal[s, a]`` give the result of
    action ``a`` in state ``states[s]``; ``start`` is the reset state. The
    worlds are deterministic, so stepping this table is the same as stepping
    the env (the step limit is the caller's job) and is much faster.
    """

    states: list
    index: dict
    next: np.ndarray
    reward: np.ndarray
    terminal: np.ndarray
    start: int

    @property
    def n_states(self) -> int:
        return len(self.states)


def transition_table(env: ChainRoomsEnv) -> TransitionTable:
    """Compile ``env`` (with its current feedback setting) into a TransitionTable."""
    states = sorted(reachable_states(env), key=repr)
    index = {s: i for i, s in enumerate(states)}
    n, n_actions = len(states), env.action_space.n
    nxt = np.zeros((n, n_actions), dtype=np.int64)
    reward = np.zeros((n, n_actions))
    terminal = np.zeros((n, n_actions), dtype=bool)
    for i, s in enumerate(states):
        if s[4] == env.n_rooms:  # terminal state: absorbing, never stepped from
            nxt[i] = i
            continue
        for a, s2, r, term in _successors(env, s):
            nxt[i, a], reward[i, a], terminal[i, a] = index[s2], r, term
    env.reset(seed=0)
    start = index[env.markov_state()]
    return TransitionTable(states, index, nxt, reward, terminal, start)
