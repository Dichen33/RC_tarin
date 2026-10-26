"""Command line interface.

Subcommands:
  run        one episode (headless or viewer)
  eval       run a suite matrix and write JSON/CSV/Markdown reports
  build      compose the MJCF scene without running a policy
  list       show registered robots, obstacles and evaluators
  selfcheck  load the checkpoint and step twice to verify the contract

For compatibility with the original sim2sim.py, flags without a subcommand
are treated as ``run``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PROJECT_ROOT, load_config, load_suite, resolve_path
from .evaluators import evaluator_names
from .obstacles import obstacle_names
from .policies import load_policy
from .robots import available_profiles
from .scene import compose_scene
from .simulation import run_episode
from .suite import run_suite

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "go2w.yaml"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML run configuration")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Policy checkpoint .pt")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--obstacle", default=None, help="Obstacle brick (flat/rough/plum/high_wall/gap/stairs/low_bar)")
    parser.add_argument("--difficulty", type=float, default=None, help="Obstacle difficulty in [0, 1]")
    parser.add_argument("--duration", type=float, default=None, help="Episode duration in seconds")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "YAW"), default=None)
    parser.add_argument("--spawn", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW"), default=None)
    parser.add_argument("--scan-size", nargs=2, type=float, metavar=("X", "Y"), default=None)
    parser.add_argument("--pass-distance", type=float, default=None, help="Override the success distance")
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="Override the MoE residual scale (PlumBlossom=0.35, Mixed=0.40)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Randomization seed")
    parser.add_argument("--randomize", action="store_true", help="Enable reset-time domain randomization")
    parser.add_argument("--scene", type=Path, default=None, help="Output composed MJCF path")
    parser.add_argument("--verbose", action="store_true", help="Print per-250-step progress")


def _apply_common(cfg, args) -> None:
    if args.checkpoint is not None:
        cfg.policy.checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.device is not None:
        cfg.policy.device = str(args.device)
    if args.obstacle is not None:
        cfg.obstacle.type = str(args.obstacle)
    if args.difficulty is not None:
        cfg.obstacle.difficulty = float(args.difficulty)
    if args.duration is not None:
        cfg.simulation.duration = float(args.duration)
    if args.command is not None:
        cfg.command = tuple(float(v) for v in args.command)
    if args.spawn is not None:
        cfg.spawn_override = tuple(float(v) for v in args.spawn)
    if args.scan_size is not None:
        cfg.policy.scan_size = (float(args.scan_size[0]), float(args.scan_size[1]))
    if args.pass_distance is not None:
        cfg.evaluation.success_distance = float(args.pass_distance)
    if args.residual_scale is not None:
        cfg.policy.residual_scale = float(args.residual_scale)
    if args.seed is not None:
        cfg.simulation.randomization.seed = int(args.seed)
    if args.randomize:
        cfg.simulation.randomization.enabled = True
    if args.scene is not None:
        cfg.output.scene_xml = Path(args.scene).expanduser().resolve()
    cfg.validate()


def _print_result(result) -> None:
    if result.result is None:
        print(f"[Result] ERROR {result.error}")
        return
    item = result.result
    metrics = item.metrics
    print(
        f"[Result] {item.status} obstacle={item.kind} reason={item.reason} "
        f"distance={item.distance:.3f}/{metrics.get('target_distance')} "
        f"hold={metrics.get('hold_steps')}/{item.features.get('metadata', {}).get('difficulty', '')} "
        f"tilt={metrics.get('max_tilt')} z={metrics.get('final_z_rel')} "
        f"illegal={metrics.get('illegal_contact')} off_course={metrics.get('off_course')} "
        f"time={item.duration:.2f}s"
    )


def _cmd_run(args) -> int:
    cfg = load_config(args.config)
    _apply_common(cfg, args)
    if args.headless and cfg.simulation.duration <= 0.0:
        raise ValueError("--duration 0 (unlimited) is only valid with the viewer; use >0 for --headless")
    if getattr(args, "course", None) is not None:
        cfg.obstacle.type = "course"
        cfg.obstacle.params["xml"] = str(Path(args.course).expanduser().resolve())
        cfg.validate()
    if args.build_only:
        scene = compose_scene(cfg, cfg.output.scene_xml)
        print(
            f"[Scene] {scene.output_xml}\n"
            f"[Scene] nq={scene.model.nq} nv={scene.model.nv} nu={scene.model.nu} "
            f"nbody={scene.model.nbody} ngeom={scene.model.ngeom}"
        )
        print(json.dumps(scene.build.summary(), indent=2))
        return 0
    policy = load_policy(cfg)
    if args.headless:
        result = run_episode(cfg, policy=policy, seed=cfg.simulation.randomization.seed, verbose=args.verbose)
        _print_result(result)
        return 0 if result.status == "PASS" else 2 if result.status == "FAIL" else 1
    from .viewer import run_viewer

    obstacle_result = run_viewer(
        cfg,
        policy,
        cfg.output.scene_xml,
        use_panel=not bool(getattr(args, "no_panel", False)),
    )
    print(
        f"[Result] {obstacle_result.status} reason={obstacle_result.reason} "
        f"distance={obstacle_result.distance:.3f} time={obstacle_result.duration:.2f}s"
    )
    return 0 if obstacle_result.status == "PASS" else 2


def _cmd_eval(args) -> int:
    base = load_config(args.config)
    _apply_common(base, args)
    suite = load_suite(args.suite)
    # CLI overrides win over suite defaults, while per-case values still win.
    if args.duration is not None:
        suite.duration = float(args.duration)
    if args.command is not None:
        suite.command = tuple(float(v) for v in args.command)
    if args.seed is not None:
        suite.seed = int(args.seed)
    out_dir = Path(args.out).expanduser().resolve() if args.out else (base.output.results_dir / suite.name)
    policy = load_policy(base)
    report = run_suite(base, suite, policy=policy, out_dir=out_dir, verbose=args.verbose)
    print(
        f"[Suite] {report.name}: {report.passed}/{report.total} PASS, "
        f"{report.failed} FAIL, {report.errors} ERROR -> {out_dir}"
    )
    print(f"[Suite] report: {out_dir / 'report.md'}")
    print(f"[Suite] results: {out_dir / 'results.json'}")
    return 0 if report.errors == 0 else 1


def _cmd_build(args) -> int:
    cfg = load_config(args.config)
    _apply_common(cfg, args)
    scene = compose_scene(cfg, cfg.output.scene_xml)
    print(
        f"[Scene] {scene.output_xml}\n"
        f"[Scene] nq={scene.model.nq} nv={scene.model.nv} nu={scene.model.nu} "
        f"nbody={scene.model.nbody} ngeom={scene.model.ngeom}"
    )
    print(json.dumps(scene.build.summary(), indent=2))
    return 0


def _cmd_list(_args) -> int:
    print("robots:    " + ", ".join(available_profiles()))
    print("obstacles: " + ", ".join(obstacle_names()))
    print("evaluators:" + ", ".join(evaluator_names()))
    return 0


def _cmd_selfcheck(args) -> int:
    cfg = load_config(args.config)
    _apply_common(cfg, args)
    policy = load_policy(cfg)
    cfg.policy.history_length = policy.contract.history_length
    cfg.policy.scan_dim = policy.contract.scan_dim
    cfg.policy.scan_size = policy.contract.scan_size
    cfg.validate()
    scene = compose_scene(cfg, cfg.output.scene_xml)
    from .evaluators import EvaluatorContext, MetricsTracker, make_evaluator
    from .observation import HeightScanner, ObservationPipeline

    scanner = HeightScanner(scene, cfg)
    observation = ObservationPipeline(scene, cfg)
    scanner.reset(0)
    observation.reset(0)
    observation.set_command(cfg.command)
    observation.update()
    scan = scanner.update()
    if scan.values.size != policy.contract.scan_dim:
        raise AssertionError("scan dimension mismatch")
    if not (0.0 <= float(scan.values.min()) <= 1.01 and 0.0 <= float(scan.values.max()) <= 1.01):
        raise AssertionError(f"scan out of range: {scan.values.min()} .. {scan.values.max()}")
    if float(scan.values.max()) < 0.05:
        raise AssertionError("all scan values are near zero; check the height-scan sign")
    proprio = observation.term_major_history()
    action = policy.act(proprio, scan.values, scanner.scan_history(policy.contract.history_length))
    if action.size != cfg.robot.action_dim:
        raise AssertionError("action dimension mismatch")
    import numpy as np

    if not (float(np.abs(action).max()) < 100.0):
        raise AssertionError("action magnitude is implausible")
    print(
        "[Selfcheck] OK\n"
        f"  scan min/mean/max = {scan.values.min():.3f}/{scan.values.mean():.3f}/{scan.values.max():.3f}\n"
        f"  scan valid fraction = {scan.valid_fraction:.3f}\n"
        f"  action min/max = {action.min():.3f}/{action.max():.3f}\n"
        f"  policy: {policy.describe()}"
    )
    return 0


def _legacy_run_parser(parser: argparse.ArgumentParser) -> None:
    _add_common(parser)
    parser.add_argument("--headless", action="store_true", help="No viewer")
    parser.add_argument("--build-only", action="store_true", help="Compose the scene and exit")
    parser.add_argument("--no-panel", action="store_true", help="Accepted for compatibility; no pygame panel")
    parser.add_argument("--course", type=Path, default=None, help="Accepted for compatibility")
    parser.add_argument("--dt", type=float, default=None, help="Unused compatibility flag")
    parser.set_defaults(func=_cmd_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sim2sim.py",
        description="Modular MuJoCo sim2sim evaluator for the HighTorque MiniDog / Go2W.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run one episode")
    _add_common(run)
    run.add_argument("--headless", action="store_true", help="No viewer")
    run.add_argument("--build-only", action="store_true", help="Compose the scene and exit")
    run.add_argument("--no-panel", action="store_true", help="Accepted for compatibility; no pygame panel")
    run.add_argument("--course", type=Path, default=None, help="Accepted for compatibility")
    run.add_argument("--dt", type=float, default=None, help="Unused compatibility flag")
    run.set_defaults(func=_cmd_run)

    evaluate = sub.add_parser("eval", help="Run a suite and write a report")
    _add_common(evaluate)
    evaluate.add_argument("--suite", type=Path, default=PROJECT_ROOT / "configs" / "suites" / "quick.yaml")
    evaluate.add_argument("--out", type=Path, default=None, help="Output directory")
    evaluate.set_defaults(func=_cmd_eval)

    build = sub.add_parser("build", help="Compose the MJCF scene and exit")
    _add_common(build)
    build.set_defaults(func=_cmd_build)

    listing = sub.add_parser("list", help="List registered bricks")
    listing.set_defaults(func=_cmd_list)

    check = sub.add_parser("selfcheck", help="Verify checkpoint, scan and action contract")
    _add_common(check)
    check.set_defaults(func=_cmd_selfcheck)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        # No arguments is the interactive path: open the v2 ROBOCON map with
        # the pygame control panel.  Close the MuJoCo window to exit.
        print("[Run] no arguments: starting the v2 ROBOCON viewer + control panel.")
        print("[Run] close the window to exit; use --headless for evaluation.")
        argv = [
            "--duration", "0",
            "--config", str(PROJECT_ROOT / "configs" / "course.yaml"),
        ]
    known = {"run", "eval", "build", "list", "selfcheck"}
    if argv[0] not in known:
        # Backward compatibility: the old CLI had no subcommands.
        parser = argparse.ArgumentParser(prog="sim2sim.py")
        _legacy_run_parser(parser)
        args = parser.parse_args(argv)
        if args.no_panel:
            print("[Legacy] --no-panel accepted; the modular runtime has no pygame panel.", file=sys.stderr)
        return int(args.func(args))
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))
