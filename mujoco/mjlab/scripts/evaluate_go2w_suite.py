"""Summarize all Go2W stage runs and optionally enforce handoff gates.

The report consumes the CSV/manifest emitted by training.  It is intentionally
separate from sim evaluation: a training metric is not presented as a physical
success rate.  Use the task-specific play/sim2sim evaluator for episode-level
success, then place its exported CSV beside the stage manifest if desired.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go2w_native.config.stage_profiles import STAGE_PROFILES


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key, "")
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


def _latest_metrics(stage_root: Path) -> tuple[Path | None, dict[str, str] | None]:
    paths = sorted(stage_root.rglob("metrics.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows:
            return path, rows[-1]
    return None, None


def _success_values(row: dict[str, str]) -> dict[str, float]:
    values = {}
    for key, raw in row.items():
        if "success_rate" not in key.lower():
            continue
        value = _float(row, key)
        if value is not None:
            values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Go2W stage metrics and transfer manifests.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "model" / "go2w_curriculum")
    parser.add_argument("--json", type=Path, default=None, help="Write the report to JSON as well as stdout.")
    parser.add_argument("--strict", action="store_true", help="Fail when a stage has no manifest or metrics.csv.")
    parser.add_argument("--enforce-gates", action="store_true", help="Require stage_evaluation.json gates.")
    parser.add_argument("--min-success", type=float, default=0.60)
    parser.add_argument("--max-fall-rate", type=float, default=0.20)
    parser.add_argument("--max-illegal-collision-rate", type=float, default=0.10)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    report: list[dict] = []
    missing = []
    for profile in STAGE_PROFILES:
        stage_root = root / profile.name
        manifest_path = stage_root / "stage_manifest.json"
        evaluation_path = stage_root / "stage_evaluation.json"
        metrics_path, row = _latest_metrics(stage_root) if stage_root.exists() else (None, None)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else None
        if manifest is None or row is None:
            missing.append(profile.name)
        item = {
            "stage": profile.name,
            "task": profile.task,
            "manifest": manifest,
            "evaluation": evaluation,
            "metrics_csv": str(metrics_path) if metrics_path else None,
            "iteration": _float(row, "iteration") if row else None,
            "mean_reward": _float(row, "mean_reward") if row else None,
            "learning_rate": _float(row, "learning_rate") if row else None,
            "success_rates": _success_values(row) if row else {},
            "policy_diagnostics": {
                key: _float(row, key)
                for key in ("Policy/router_entropy", "Policy/residual_norm", "Policy/wheel_usage_ema", "Policy/step_usage_ema", "Policy/gap_usage_ema", "Policy/recovery_usage_ema")
                if row and _float(row, key) is not None
            },
        }
        report.append(item)
        status = "missing" if manifest is None or row is None else "ok"
        if args.enforce_gates:
            if evaluation is None:
                status = "gate_missing"
                missing.append(f"{profile.name}:stage_evaluation.json")
            else:
                success = float(evaluation.get("success_rate", -1.0))
                fall = float(evaluation.get("fall_rate", 1.0))
                collision = float(evaluation.get("illegal_collision_rate", 1.0))
                if success < args.min_success or fall > args.max_fall_rate or collision > args.max_illegal_collision_rate:
                    status = "gate_failed"
                    missing.append(f"{profile.name}:gate")
        print(f"{profile.name}: {status}; metrics={metrics_path or '-'}; success_rates={item['success_rates'] or '-'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if missing:
        print(f"missing_stages={','.join(missing)}")
    if args.strict and missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

