"""Roll out a random policy in any Minigrid / Miniworld env and report what happened.

Examples (from the repo root):

    python scripts/run_env.py --env MiniGrid-Proj-GoToObj-v0 --episodes 3 --seed 0
    python scripts/run_env.py --env MiniWorld-Proj-GoToObj-v0 --obs dict --save-frames outputs/goto3d
    python scripts/run_env.py --env MiniGrid-Proj-KeyDoor-v0 --kw size=10 randomize=True --render human

Prints JSON: env/obs spaces, then per-episode return, step count,
terminated/truncated flags and mission. ``--save-frames DIR`` writes the
first episode as PNG frames plus an animated ``episode.gif``.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mini_envs import library_of, make_env  # noqa: E402


def parse_kwargs(pairs: list[str]) -> dict:
    """Parse ``key=value`` strings; values go through ``ast.literal_eval`` when possible."""
    out = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise SystemExit(f"--kw expects key=value, got {pair!r}")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def save_frames(frames: list[np.ndarray], out_dir: Path) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    for i, img in enumerate(images):
        img.save(out_dir / f"frame_{i:04d}.png")
    images[0].save(
        out_dir / "episode.gif", save_all=True, append_images=images[1:], duration=150, loop=0
    )


def run(args: argparse.Namespace) -> dict:
    render_mode = "human" if args.render == "human" else ("rgb_array" if args.save_frames else None)
    env = make_env(
        args.env,
        obs=args.obs,
        max_steps=args.max_steps,
        render_mode=render_mode,
        **parse_kwargs(args.kw),
    )
    rng = np.random.default_rng(args.seed)
    episodes, frames = [], []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            record = args.save_frames and ep == 0
            if record:
                frames.append(env.render())
            total, steps = 0.0, 0
            while True:
                action = int(rng.integers(env.action_space.n))
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                steps += 1
                if record:
                    frames.append(env.render())
                if terminated or truncated:
                    break
            episodes.append(
                dict(
                    episode=ep,
                    return_=total,
                    steps=steps,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    mission=getattr(env.unwrapped, "mission", None),
                )
            )
    finally:
        env.close()

    if frames:
        save_frames(frames, Path(args.save_frames))

    return dict(
        env=args.env,
        library=library_of(args.env),
        obs_mode=args.obs,
        observation_space=str(env.observation_space),
        action_space=str(env.action_space),
        seed=args.seed,
        episodes=episodes,
        frames_dir=args.save_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="MiniGrid-Proj-GoToObj-v0")
    parser.add_argument("--obs", default=None, help="observation mode (see mini_envs.OBS_MODES)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render", choices=["none", "human"], default="none")
    parser.add_argument("--save-frames", default=None, metavar="DIR")
    parser.add_argument("--kw", nargs="*", default=[], metavar="KEY=VALUE", help="extra env constructor kwargs")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
