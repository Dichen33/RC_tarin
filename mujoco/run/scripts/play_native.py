from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


MJLAB_ROOT = Path(__file__).resolve().parents[1]
if str(MJLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(MJLAB_ROOT))


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _find_run_dir(run_name: str) -> Path:
    root = MJLAB_ROOT / "model" / run_name
    if not root.exists():
        raise FileNotFoundError(f"Run folder not found: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        return root
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint.endswith(".pt"):
        path = run_dir / checkpoint
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt found in {run_dir}")
    if checkpoint == "latest":
        return checkpoints[-1]
    path = run_dir / f"{checkpoint}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Play MiniDog native MJLab/RSL-RL checkpoint from model/<run_name>.")
    parser.add_argument("task", help="Robot-Flat-v0, Robot-Rough-v0, or Robot-Crawl-v0")
    parser.add_argument("--run-name", required=True, help="Folder under model, for example rough_v1")
    parser.add_argument("--checkpoint", default="latest", help="latest, model_9999.pt, or model_9999")
    args, extra = parser.parse_known_args()

    run_dir = _find_run_dir(args.run_name)
    checkpoint = _find_checkpoint(run_dir, args.checkpoint)

    from mjlab.scripts.play import main as mjlab_play_main

    if extra and extra[0] == "--":
        extra = extra[1:]
    sys.argv = ["play", args.task, "--checkpoint-file", str(checkpoint), *extra]
    print(f"[MiniDog] loading checkpoint: {checkpoint}")
    mjlab_play_main()


if __name__ == "__main__":
    main()
