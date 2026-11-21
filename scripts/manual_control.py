"""Play any Minigrid / Miniworld env with the keyboard, to sanity-check a new layout.

    python scripts/manual_control.py --env MiniGrid-Proj-KeyDoor-v0
    python scripts/manual_control.py --env MiniWorld-Proj-GoToObj-v0 --top-view

Minigrid keys: arrows = turn/forward, space = toggle, pageup/pagedown =
pickup/drop, enter = done, backspace = reset, escape = quit.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="MiniGrid-Proj-KeyDoor-v0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-view", action="store_true", help="Miniworld only: top-down camera")
    parser.add_argument("--no-time-limit", action="store_true", help="Miniworld only")
    args = parser.parse_args()

    if library_of(args.env) == "minigrid":
        from minigrid.manual_control import ManualControl

        env = make_env(args.env, render_mode="human")
        ManualControl(env, seed=args.seed).start()
    else:
        from miniworld.manual_control import ManualControl

        env = make_env(args.env, render_mode="human", view="top" if args.top_view else "agent")
        ManualControl(env, no_time_limit=args.no_time_limit, domain_rand=False).run()


if __name__ == "__main__":
    main()
