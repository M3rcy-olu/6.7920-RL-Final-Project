"""List registered Minigrid / Miniworld env ids.

    python scripts/list_envs.py              # everything (~200 ids)
    python scripts/list_envs.py --custom     # only ours (mini_envs/)
    python scripts/list_envs.py --library miniworld
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_envs import list_env_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", choices=["minigrid", "miniworld"], default=None)
    parser.add_argument("--custom", action="store_true", help="only envs defined in mini_envs/")
    args = parser.parse_args()
    for env_id in list_env_ids(args.library, custom_only=args.custom):
        print(env_id)


if __name__ == "__main__":
    main()
