"""Train one unified Go2W policy in a single PPO run.

The Mixed task contains the complete terrain family, release gates, and the
bounded LP-ACRL terrain/level sampler. This wrapper deliberately does not load
a previous stage checkpoint: one actor, one critic, one optimizer, and one
continuous curriculum are used from iteration zero. The curriculum starts
with flat/rough competence and releases stairs, gaps, walls, and stepping
stones by measured success and learning progress.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-run unified Go2W terrain training.")
    parser.add_argument("--iterations", type=int, default=132000, help="Total PPO updates in this one run.")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--log-root", type=Path, default=ROOT / "model" / "go2w_unified")
    parser.add_argument("--run-name", default="go2w_unified")
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument(
        "--ccd-iterations",
        type=int,
        default=8,
        help=(
            "Training-only MuJoCo CCD iterations; play/sim2sim always keeps the "
            "nominal value (80). CCD runs on every terrain pair each substep and "
            "was the second largest per-iteration cost at the old default of 80."
        ),
    )
    parser.add_argument("--upload-model", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER)
    # A bare ``--`` is the conventional "pass the rest through" marker, but
    # argparse treats it as end-of-options and then rejects the wrapped trainer's
    # flags (--stage-checkpoint ...).  Accept it as a synonym for ``--extra``.
    argv = sys.argv[1:]
    if "--extra" not in argv and "--" in argv:
        index = argv.index("--")
        argv = argv[:index] + ["--extra"] + argv[index + 1:]
    args = parser.parse_args(argv)

    if args.ccd_iterations is not None:
        if args.ccd_iterations < 1:
            parser.error("--ccd-iterations must be positive")
        # The environment config reads this only for non-play training tasks.
        import os

        os.environ["MINIDOG_TRAIN_CCD_ITERATIONS"] = str(args.ccd_iterations)

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_native.py"),
        "Robot-Go2W-Mixed-v0",
        "--env.scene.num-envs",
        str(args.num_envs),
        "--agent.max-iterations",
        str(args.iterations),
        "--agent.run-name",
        args.run_name,
        "--agent.logger",
        args.logger,
        "--agent.upload-model",
        str(args.upload_model),
        "--log-root",
        str(args.log_root.expanduser().resolve()),
    ]
    if args.extra:
        command.extend(args.extra[1:] if args.extra[:1] == ["--"] else args.extra)
    print("[Go2W unified]", " ".join(f'"{part}"' if " " in part else part for part in command))
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

