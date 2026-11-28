"""Play any Minigrid / Miniworld env with the keyboard, to sanity-check a new layout.

    python scripts/manual_control.py --env MiniGrid-Proj-KeyDoor-v0
    python scripts/manual_control.py --env MiniWorld-Proj-GoToObj-v0 --top-view
    python scripts/manual_control.py --env MiniGrid-Proj-Chain-Base-v0 --feedback-every end

Minigrid keys: arrows = turn/forward, space = toggle, pageup/pagedown =
pickup/drop, enter = done, backspace = reset, escape = quit.
Chain worlds: same keys, but drop and done do nothing (no such actions).
Miniworld keys: arrows = move/turn, P = pickup, D = drop, enter = done,
backspace = reset, escape = quit.

This needs a display; it opens a pygame (Minigrid) or pyglet (Miniworld) window.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_envs import library_of, make_env  # noqa: E402


class _ChainKeys:
    """Translate Minigrid's keyboard actions to the chain worlds' 5-action set.

    Drop and done don't exist there, so their keys do nothing. After every step
    it prints the room the agent is in, what it carries and the mistake flag.
    """

    def __init__(self, env):
        from minigrid.core.actions import Actions

        self.env = env
        self._index = {a: i for i, a in enumerate(env.unwrapped._action_map)}
        self._actions = Actions

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        if action not in self._index:
            print(f"{self._actions(action).name} is not an action in chain worlds")
            return None, 0.0, False, False, {}
        out = self.env.step(self._index[action])
        u = self.env.unwrapped
        carried = u.carrying.color if u.carrying is not None else "-"
        print(f"  room={u.frontier} carrying={carried} mistake={u.mistake} limit={u.max_steps}")
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="MiniGrid-Proj-KeyDoor-v0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-view", action="store_true", help="Miniworld only: top-down camera")
    parser.add_argument("--no-time-limit", action="store_true", help="Miniworld only")
    parser.add_argument(
        "--feedback-every",
        default=None,
        help="Chain worlds only: pay reward every k rooms (1, 2, ...) or 'end'",
    )
    args = parser.parse_args()

    if library_of(args.env) == "minigrid":
        from minigrid.manual_control import ManualControl

        kwargs = {}
        if args.env.startswith("MiniGrid-Proj-Chain-") and args.feedback_every is not None:
            kwargs["feedback_every"] = None if args.feedback_every == "end" else int(args.feedback_every)
        env = make_env(args.env, render_mode="human", **kwargs)
        if hasattr(env.unwrapped, "markov_state"):  # chain-of-rooms worlds use 5 actions
            env = _ChainKeys(env)
        ManualControl(env, seed=args.seed).start()
    else:
        from miniworld.manual_control import ManualControl

        env = make_env(args.env, render_mode="human", view="top" if args.top_view else "agent")
        ManualControl(env, no_time_limit=args.no_time_limit, domain_rand=False).run()


if __name__ == "__main__":
    main()
