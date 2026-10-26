"""Play / record the Go2W plum-blossom policy on a chosen difficulty row.

The stock ``python -m mjlab.scripts.play`` always spawns at terrain level 0
(``max_init_terrain_level=0`` in the env cfg) and only one of the four terrain
columns is the stepping-stone course, so it shows the *easiest* row at best.
This script pins the row (and the column) before the reset, so you can watch the
competition-size course (level 6, 0.200 m cubes) directly.

Examples
--------
# live window (Windows / any machine with a display)
python scripts/play_plum.py --checkpoint D:/models/model_3496.pt --level 6

# record an mp4 (works headless too)
python scripts/play_plum.py --checkpoint /root/run/model/.../model_3496.pt \
    --level 6 --num-envs 4 --video --steps 1200 --out play_L6.mp4

# walk every row in one go: run it 7 times with --level 0..6
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TASK_ID = "Robot-Go2W-Mixed-v0"
PLUM_TERRAIN = "rough_plum_blossom"

# Importing the task package is what registers ``Robot-Go2W-*`` in mjlab's
# registry; without it ``load_env_cfg`` raises ``KeyError: Robot-Go2W-Mixed-v0``.
# ``mjlab.scripts.play`` gets this as a side effect of its own imports, a small
# script has to do it explicitly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import go2w_native  # noqa: E402,F401  (registers the Robot-Go2W-* tasks)


def build_env(args, device):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    # The default camera is ORIGIN_TYPE.AUTO, which resolves to a *free* camera at
    # ``lookat=(0,0,0)``.  Our pinned patch sits at x ~ 22 m, so the robot shows up
    # as a dot in the distance.  Track the robot base instead.
    from mjlab.viewer.viewer_config import ViewerConfig

    env_cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    env_cfg.viewer.entity_name = 'robot'
    env_cfg.viewer.distance = args.cam_distance
    env_cfg.viewer.elevation = args.cam_elevation
    env_cfg.viewer.azimuth = args.cam_azimuth
    # ``play=True`` deliberately shrinks the terrain grid (5 rows, curriculum off)
    # because the stock player only ever spawns on row 0.  We want the *training*
    # ladder back: one column per terrain type and one row per difficulty level, so
    # that "level 6" means the same patch it meant during training.
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is not None:
        generator.curriculum = True
        generator.num_rows = args.rows
        generator.num_cols = len(generator.sub_terrains)
        print(
            f"[play_plum] terrain grid restored: rows={generator.num_rows} "
            f"cols={generator.num_cols} curriculum=True"
        )
    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("This task has no curriculum terrain to pin a level on.")
    names = list(env_cfg.scene.terrain.terrain_generator.sub_terrains.keys())
    if PLUM_TERRAIN not in names:
        raise RuntimeError(
            f"{PLUM_TERRAIN} is not in this task's terrain columns: {names}. "
            "Did you forget MINIDOG_OBSTACLES=plum ?"
        )
    column = names.index(PLUM_TERRAIN)
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    level = max(0, min(args.level, max_level))

    env_ids = torch.arange(args.num_envs, device=device)
    terrain.terrain_types[env_ids] = column
    terrain.terrain_levels[env_ids] = level
    # The spawn position is read from env_origins, so both tensors must move.
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
    print(
        f"[play_plum] column={column} ({PLUM_TERRAIN})  level={level}  "
        f"num_envs={args.num_envs}  render_mode={render_mode}"
    )
    return env, level


def build_policy(env, args, device):
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    agent_cfg = load_rl_cfg(TASK_ID)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    print(f"[play_plum] loaded {checkpoint.name}")
    return wrapped, runner.get_inference_policy(device=device)


def record(wrapped_env, policy, args):
    import mediapy as media

    obs, _ = wrapped_env.reset()
    frames = []
    with torch.no_grad():
        for step in range(args.steps):
            actions = policy(obs)
            obs, _, dones, _ = wrapped_env.step(actions)
            frame = wrapped_env.unwrapped.render()
            if frame is not None:
                frames.append(frame)
            if bool(torch.as_tensor(dones).any()) and args.stop_on_done:
                print(f"[play_plum] episode ended at step {step}; stopping")
                break
    if not frames:
        raise RuntimeError("No frames were rendered; is render_mode wired up?")
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(out), frames, fps=args.fps)
    print(f"[play_plum] wrote {out}  ({len(frames)} frames @ {args.fps} fps)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="model_XXXX.pt to load")
    parser.add_argument("--level", type=int, default=6, help="terrain row to pin (0..6, 6 = competition)")
    parser.add_argument("--rows", type=int, default=7, help="curriculum rows to build (training used 7)")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    parser.add_argument("--video", action="store_true", help="record an mp4 instead of opening a window")
    parser.add_argument("--steps", type=int, default=1500, help="frames to record (video mode)")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--out", default="play_plum.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cam-distance", type=float, default=2.6, help="camera distance to the robot")
    parser.add_argument("--cam-elevation", type=float, default=-18.0, help="camera elevation (deg)")
    parser.add_argument("--cam-azimuth", type=float, default=90.0, help="camera azimuth (deg); 90 = side view")
    parser.add_argument("--stop-on-done", action="store_true", help="stop after the first episode ends")
    args = parser.parse_args()

    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env, _level = build_env(args, device)
    wrapped, policy = build_policy(env, args, device)

    if args.video:
        record(wrapped, policy, args)
        env.close()
        return 0

    from mjlab.viewer import NativeMujocoViewer

    NativeMujocoViewer(wrapped, policy).run()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
