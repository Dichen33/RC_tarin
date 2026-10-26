from __future__ import annotations

import sys
import importlib
import os
from pathlib import Path


MJLAB_ROOT = Path(__file__).resolve().parents[1]
if str(MJLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(MJLAB_ROOT))


TASK_TO_RUN_NAME = {
    "Robot-Flat-v0": "flat",
    "Robot-Rough-v0": "rough",
    "Robot-Crawl-v0": "crawl",
    "Robot-PlumBlossom-v0": "plum_blossom",
    "Robot-Go2W-Flat-v0": "go2w_flat",
    "Robot-Go2W-Rough-v0": "go2w_rough",
    "Robot-Go2W-Step-v0": "go2w_step",
    "Robot-Go2W-PlumBlossom-v0": "go2w_plum_blossom",
    "Robot-Go2W-Gap-v0": "go2w_gap",
    "Robot-Go2W-HighWall-v0": "go2w_high_wall",
    "Robot-Go2W-Mixed-v0": "go2w_mixed",
}


def _configure_local_warp_cache() -> None:
    """Keep Warp's generated kernels in the writable project workspace."""
    try:
        import warp as wp

        cache_dir = MJLAB_ROOT / ".warp_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(cache_dir)
    except Exception:
        # Warp is initialized by MJLab later.  If it is unavailable here, the
        # normal MJLab error remains more useful than hiding the import error.
        pass


def _has_arg(args: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


def _get_arg_value(args: list[str], name: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return None


def _default_run_name(task: str) -> str:
    return TASK_TO_RUN_NAME.get(task, task.replace("/", "_").replace(":", "_"))


def _extract_stage_options(args: list[str]) -> tuple[list[str], dict[str, str]]:
    """Remove native stage flags before MJLab's tyro parser sees them."""
    value_options = {"--stage-checkpoint", "--stage-transfer", "--stage-source-stage"}
    cleaned: list[str] = []
    options: dict[str, str] = {}
    index = 0
    while index < len(args):
        arg = args[index]
        matched = next((name for name in value_options if arg == name or arg.startswith(f"{name}=")), None)
        if matched is None:
            cleaned.append(arg)
            index += 1
            continue
        if "=" in arg:
            value = arg.split("=", 1)[1]
        elif index + 1 < len(args):
            value = args[index + 1]
            index += 1
        else:
            raise SystemExit(f"{matched} requires a value")
        options[matched] = value
        index += 1
    return cleaned, options


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("usage: train_native.py <TASK> [MJLab train options]")
        print("")
        print("Example:")
        print(
            "  uv run python scripts/train_native.py Robot-Rough-v0 "
            "--env.scene.num-envs 2048 --agent.max-iterations 15000 "
            "--agent.run-name rough_v1 --agent.logger tensorboard --agent.upload-model False"
        )
        raise SystemExit(0)

    task = sys.argv[1]
    args, stage_options = _extract_stage_options(sys.argv[2:])

    stage_checkpoint = stage_options.get("--stage-checkpoint")
    stage_transfer = stage_options.get("--stage-transfer", "actor_critic_reset_optimizer")
    valid_transfers = {
        "full_resume",
        "actor_critic_reset_optimizer",
        "actor_only_reset_critic_optimizer",
    }
    if stage_transfer not in valid_transfers:
        raise SystemExit(f"--stage-transfer must be one of: {', '.join(sorted(valid_transfers))}")
    if stage_checkpoint:
        checkpoint = Path(stage_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage checkpoint not found: {checkpoint}")
        os.environ["MINIDOG_STAGE_TRANSFER_MODE"] = stage_transfer
        os.environ["MINIDOG_STAGE_SOURCE_STAGE"] = stage_options.get("--stage-source-stage", "")
        resume_value = _get_arg_value(args, "--agent.resume")
        if resume_value is not None and resume_value.lower() in {"false", "0", "no"}:
            raise SystemExit("--stage-checkpoint requires --agent.resume=True")
        if not _has_arg(args, "--agent.resume"):
            args += ["--agent.resume", "True"]

    _configure_local_warp_cache()

    # MJLab's CLI discovers built-in tasks itself, but local native packages
    # must be imported explicitly so their registration hooks run before the
    # argument parser builds its task choices.
    importlib.import_module("minidog_native")
    importlib.import_module("go2w_native")

    run_name = _get_arg_value(args, "--agent.run-name") or _default_run_name(task)

    injected: list[str] = []
    if not _has_arg(args, "--agent.run-name"):
        injected += ["--agent.run-name", run_name]
    if not _has_arg(args, "--log-root"):
        injected += ["--log-root", str(MJLAB_ROOT / "model" / run_name)]
    if not _has_arg(args, "--agent.experiment-name"):
        injected += ["--agent.experiment-name", "."]

    train_module = importlib.import_module("mjlab.scripts.train")
    if stage_checkpoint:
        # The upstream CLI resolves resume files by regex. Replace only that
        # resolver for this explicit path; other train behavior is unchanged.
        train_module.get_checkpoint_path = lambda *_args, **_kwargs: checkpoint

    sys.argv = ["train", task, *args, *injected]
    print(f"[MiniDog] model output root: {MJLAB_ROOT / 'model' / run_name}")
    if stage_checkpoint:
        print(f"[MiniDog] stage transfer: {stage_transfer} <- {checkpoint}")
    train_module.main()


if __name__ == "__main__":
    main()
