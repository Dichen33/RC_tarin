from __future__ import annotations

import argparse
import sys
from pathlib import Path


MJLAB_ROOT = Path(__file__).resolve().parents[2]
if str(MJLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(MJLAB_ROOT))

from minidog_rl.scripts.eval_policy_pt import main as eval_policy_pt_main
from minidog_rl.scripts.eval_sb3 import main as eval_sb3_main
from minidog_rl.paths import LEGACY_LOGS_ROOT, MODEL_ROOT, RUNS_ROOT
from minidog_rl.tasks import LEGACY_TASK, resolve_task_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacGym-style play entry for the MJLab MiniDog policy.")
    parser.add_argument("--task", type=str, default="Robot-Flat-v0")
    parser.add_argument("--experiment_name", "--experiment-name", dest="experiment_name", type=str, default=None)
    parser.add_argument("--load_run", "--load-run", dest="load_run", type=str, default="residual_v1")
    parser.add_argument("--checkpoint", type=str, default="final_policy")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.09)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--terrain", type=str, default=None, choices=["plane", "rough"])
    parser.add_argument(
        "--terrain-profile",
        "--terrain_profile",
        dest="terrain_profile",
        type=str,
        default=None,
        choices=["mixed", "slope_up", "slope_down", "stairs_up", "stairs_down", "wall", "blocks"],
    )
    parser.add_argument("--terrain-level", "--terrain_level", dest="terrain_level", type=int, default=None)
    parser.add_argument("--terrain-nrow", "--terrain_nrow", dest="terrain_nrow", type=int, default=128)
    parser.add_argument("--terrain-ncol", "--terrain_ncol", dest="terrain_ncol", type=int, default=128)
    parser.add_argument("--terrain-width-m", "--terrain_width_m", dest="terrain_width_m", type=float, default=4.0)
    parser.add_argument("--terrain-length-m", "--terrain_length_m", dest="terrain_length_m", type=float, default=4.0)
    parser.add_argument("--render", dest="render", action="store_true", default=True)
    parser.add_argument("--headless", "--no-render", dest="render", action="store_false")
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    task_cfg = resolve_task_cfg(args.task)
    experiment_name = args.experiment_name or task_cfg.experiment_name
    if args.model:
        return Path(args.model)

    raw_name = args.checkpoint
    names = [raw_name] if Path(raw_name).suffix else [f"{raw_name}.pt", f"{raw_name}.zip"]
    roots = [
        MODEL_ROOT / experiment_name / args.load_run,
        RUNS_ROOT / experiment_name / args.load_run,
        RUNS_ROOT / experiment_name / args.load_run / "model",
        LEGACY_LOGS_ROOT / experiment_name / args.load_run,
        LEGACY_LOGS_ROOT / experiment_name / args.load_run / "model",
        MJLAB_ROOT / "runs" / experiment_name / args.load_run,
        MJLAB_ROOT / "runs" / args.load_run,
    ]
    for root in roots:
        for name in names:
            path = root / name
            if path.exists():
                return path
    return roots[0] / names[0]


def main() -> None:
    args = parse_args()
    task_cfg = resolve_task_cfg(args.task)
    model_path = resolve_model_path(args)
    sys.argv = [
        "eval_sb3",
        "--task",
        task_cfg.task_id,
        "--model",
        str(model_path),
        "--steps",
        str(args.steps),
        "--vx",
        str(args.vx),
        "--vy",
        str(args.vy),
        "--yaw",
        str(args.yaw),
        "--terrain",
        str(args.terrain or task_cfg.terrain_mode),
        "--terrain-profile",
        str(args.terrain_profile or task_cfg.terrain_profile),
        "--terrain-level",
        str(task_cfg.terrain_level if args.terrain_level is None else args.terrain_level),
        "--terrain-nrow",
        str(args.terrain_nrow),
        "--terrain-ncol",
        str(args.terrain_ncol),
        "--terrain-width-m",
        str(args.terrain_width_m),
        "--terrain-length-m",
        str(args.terrain_length_m),
    ]
    if args.render:
        sys.argv.append("--render")
    if model_path.suffix.lower() == ".pt":
        eval_policy_pt_main()
    else:
        eval_sb3_main()


if __name__ == "__main__":
    main()
