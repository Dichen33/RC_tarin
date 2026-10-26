"""Run the Go2W stages in order with explicit checkpoint transfer.

This is an orchestration script. It starts one normal ``train_native.py``
process per stage, so each stage has an independent optimizer and log folder.
Use ``--dry-run`` first; existing stage output is never reused unless its
checkpoint is explicitly selected or ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go2w_native.config.stage_profiles import STAGE_PROFILES


def _latest_checkpoint(stage_root: Path) -> Path | None:
    checkpoints = [path for path in stage_root.rglob("model_*.pt") if path.is_file()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def _manifest_path(stage_root: Path) -> Path:
    return stage_root / "stage_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Go2W Flat -> Rough -> Step -> Gap -> Wall -> Plum -> Mixed.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "model" / "go2w_curriculum")
    parser.add_argument("--iterations", type=int, default=None, help="Override max_iterations for every stage.")
    parser.add_argument("--start-stage", choices=tuple(profile.name for profile in STAGE_PROFILES), default="flat")
    parser.add_argument("--from-checkpoint", type=Path, default=None, help="Source checkpoint for --start-stage when it is not flat.")
    parser.add_argument("--overwrite", action="store_true", help="Allow adding a new run under an existing stage folder.")
    parser.add_argument("--require-evaluation", action="store_true", help="Require stage_evaluation.json before advancing.")
    parser.add_argument("--min-success", type=float, default=0.60)
    parser.add_argument("--max-fall-rate", type=float, default=0.20)
    parser.add_argument("--max-illegal-collision-rate", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra arguments forwarded to train_native.py.")
    args = parser.parse_args()

    start_index = next(index for index, profile in enumerate(STAGE_PROFILES) if profile.name == args.start_stage)
    if start_index > 0 and args.from_checkpoint is None:
        raise SystemExit("--from-checkpoint is required when --start-stage is not flat")
    if args.from_checkpoint is not None and not args.from_checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.from_checkpoint}")

    output_root = args.output_root.expanduser().resolve()
    previous_checkpoint = args.from_checkpoint.expanduser().resolve() if args.from_checkpoint else None
    previous_stage = "external" if previous_checkpoint else ""
    for profile in STAGE_PROFILES[start_index:]:
        stage_root = output_root / profile.name
        existing = _latest_checkpoint(stage_root) if stage_root.exists() else None
        if existing is not None and not args.overwrite:
            raise SystemExit(
                f"Existing checkpoint found for stage '{profile.name}': {existing}. "
                "Pass --overwrite or choose --from-checkpoint explicitly."
            )

        iterations = args.iterations or profile.max_iterations
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_native.py"),
            profile.task,
            "--log-root",
            str(stage_root),
            "--agent.run-name",
            profile.name,
            "--agent.max-iterations",
            str(iterations),
            "--agent.algorithm.learning-rate",
            str(profile.learning_rate),
            "--agent.logger",
            "tensorboard",
            "--agent.upload-model",
            "False",
        ]
        if previous_checkpoint is not None:
            command += [
                "--stage-checkpoint",
                str(previous_checkpoint),
                "--stage-transfer",
                profile.transfer_mode,
                "--stage-source-stage",
                previous_stage,
            ]
        if args.extra:
            command.extend(args.extra[1:] if args.extra[:1] == ["--"] else args.extra)

        print("[Go2W curriculum]", " ".join(f'"{part}"' if " " in part else part for part in command))
        if args.dry_run:
            previous_stage = profile.name
            continue

        stage_root.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            return completed.returncode
        checkpoint = _latest_checkpoint(stage_root)
        if checkpoint is None:
            raise SystemExit(f"Stage '{profile.name}' finished without a model_*.pt checkpoint")
        manifest = {
            "stage": profile.name,
            "task": profile.task,
            "source_checkpoint": str(previous_checkpoint) if previous_checkpoint else None,
            "source_stage": previous_stage or None,
            "transfer_mode": profile.transfer_mode,
            "critic_transfer": profile.critic_transfer,
            "history_revisit": profile.history_revisit,
            "stage_iteration": 0,
            "global_parent_stage": previous_stage or None,
            "max_iterations": iterations,
            "learning_rate": profile.learning_rate,
            "checkpoint": str(checkpoint),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _manifest_path(stage_root).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if args.require_evaluation:
            evaluation_path = stage_root / "stage_evaluation.json"
            if not evaluation_path.is_file():
                raise SystemExit(f"Stage '{profile.name}' has no {evaluation_path.name}; refusing to advance")
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            success = float(evaluation.get("success_rate", -1.0))
            fall = float(evaluation.get("fall_rate", 1.0))
            collision = float(evaluation.get("illegal_collision_rate", 1.0))
            if success < args.min_success or fall > args.max_fall_rate or collision > args.max_illegal_collision_rate:
                raise SystemExit(
                    f"Stage '{profile.name}' failed gate: success={success:.3f}, "
                    f"fall={fall:.3f}, illegal_collision={collision:.3f}"
                )
        previous_checkpoint = checkpoint
        previous_stage = profile.name
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

