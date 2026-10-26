"""Batch evaluation matrix and human-readable reports."""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RunConfig, SuiteConfig
from .policies import PolicyBackend, load_policy
from .simulation import EpisodeResult, run_episode


@dataclass
class SuiteReport:
    name: str
    results: list[EpisodeResult]
    out_dir: Path
    policy_summary: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for item in self.results if item.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for item in self.results if item.status == "FAIL")

    @property
    def errors(self) -> int:
        return sum(1 for item in self.results if item.status == "ERROR")

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": round(self.pass_rate, 5),
            "policy": self.policy_summary,
            "extras": self.extras,
            "episodes": [item.as_dict() for item in self.results],
        }


def _case_cfg(base: RunConfig, suite: SuiteConfig, case, repeat: int) -> RunConfig:
    cfg = copy.deepcopy(base)
    cfg.obstacle = copy.deepcopy(case.obstacle)
    if case.command is not None:
        cfg.command = tuple(float(v) for v in case.command)
    elif suite.command is not None:
        cfg.command = tuple(float(v) for v in suite.command)
    if case.duration is not None:
        cfg.simulation.duration = float(case.duration)
    if suite.duration is not None:
        cfg.simulation.duration = float(suite.duration)
    if case.evaluation is not None:
        cfg.evaluation = copy.deepcopy(case.evaluation)
    if suite.randomization is not None:
        cfg.simulation.randomization = copy.deepcopy(suite.randomization)
    cfg.simulation.randomization.seed = int(suite.seed + repeat)
    cfg.validate()
    return cfg


def run_suite(
    base_cfg: RunConfig,
    suite: SuiteConfig,
    policy: PolicyBackend | None = None,
    out_dir: str | Path | None = None,
    verbose: bool = False,
) -> SuiteReport:
    out_dir = Path(out_dir) if out_dir is not None else (base_cfg.output.results_dir / suite.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if policy is None:
        policy = load_policy(base_cfg)
    results: list[EpisodeResult] = []
    started = time.perf_counter()
    total = sum((case.repeats or suite.repeats) for case in suite.cases)
    index = 0
    for case in suite.cases:
        repeats = int(case.repeats or suite.repeats)
        for repeat in range(repeats):
            index += 1
            cfg = _case_cfg(base_cfg, suite, case, repeat)
            scene_xml = out_dir / "scenes" / f"scene_{index:03d}_{cfg.obstacle.type}_d{cfg.obstacle.difficulty:.2f}.xml"
            cfg.output.scene_xml = scene_xml
            cfg.output.results_dir = out_dir / "traces"
            cfg.output.save_trace = False
            cfg.output.save_failed_trace = False
            label = case.label or cfg.obstacle.type
            print(
                f"[Suite] {index}/{total} {label} d={cfg.obstacle.difficulty:.2f} "
                f"repeat={repeat} seed={cfg.simulation.randomization.seed}",
                flush=True,
            )
            result = run_episode(
                cfg,
                policy=policy,
                seed=cfg.simulation.randomization.seed,
                scene_xml=scene_xml,
                verbose=verbose,
            )
            result.extras["label"] = label
            result.extras["case_index"] = index
            result.extras["repeat"] = repeat
            results.append(result)
            status = result.status
            reason = "" if result.result is None else result.result.reason
            print(f"[Suite]   -> {status} ({reason})", flush=True)
    elapsed = time.perf_counter() - started
    report = SuiteReport(
        name=suite.name,
        results=results,
        out_dir=out_dir,
        policy_summary=policy.describe(),
        extras={
            "elapsed_s": round(elapsed, 3),
            "cases": len(suite.cases),
            "repeats": suite.repeats,
            "duration": suite.duration or base_cfg.simulation.duration,
            "robot": base_cfg.robot.name,
            "checkpoint": None if base_cfg.policy.checkpoint is None else str(base_cfg.policy.checkpoint),
        },
    )
    write_reports(report)
    return report


def write_reports(report: SuiteReport) -> None:
    out = report.out_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for item in report.results:
        result = item.result.as_dict() if item.result is not None else {}
        cfg = item.config
        obstacle = cfg.get("obstacle", {})
        rows.append(
            {
                "label": item.extras.get("label", obstacle.get("type", "")),
                "obstacle": obstacle.get("type", ""),
                "difficulty": obstacle.get("difficulty", ""),
                "status": item.status,
                "reason": result.get("reason", item.error or ""),
                "distance": result.get("distance", 0.0),
                "target": result.get("metrics", {}).get("target_distance", ""),
                "time": item.time,
                "steps": item.steps,
                "seed": item.seed,
                "scene": item.scene_xml,
                "trace": item.trace_path or "",
            }
        )
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["label"])
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate by obstacle type and difficulty.
    groups: dict[tuple[str, float], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["obstacle"]), float(row["difficulty"] or 0.0))
        group = groups.setdefault(key, {"pass": 0, "fail": 0, "error": 0, "distances": []})
        if row["status"] == "PASS":
            group["pass"] += 1
        elif row["status"] == "FAIL":
            group["fail"] += 1
        else:
            group["error"] += 1
        group["distances"].append(float(row["distance"] or 0.0))

    lines = [
        f"# sim2sim evaluation: {report.name}",
        "",
        f"- robot: `{report.extras.get('robot')}`",
        f"- checkpoint: `{report.extras.get('checkpoint')}`",
        f"- policy: {report.policy_summary}",
        f"- episodes: {report.total} (PASS {report.passed}, FAIL {report.failed}, ERROR {report.errors}), "
        f"pass rate {100.0 * report.pass_rate:.1f}%",
        f"- duration per episode: {report.extras.get('duration')} s",
        f"- wall time: {report.extras.get('elapsed_s')} s",
        "",
        "## Aggregate",
        "",
        "| obstacle | difficulty | PASS | FAIL | ERROR | pass rate | mean distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (kind, difficulty), group in sorted(groups.items()):
        total = group["pass"] + group["fail"] + group["error"]
        mean_distance = sum(group["distances"]) / len(group["distances"]) if group["distances"] else 0.0
        lines.append(
            f"| {kind} | {difficulty:.2f} | {group['pass']} | {group['fail']} | "
            f"{group['error']} | {100.0 * group['pass'] / max(total, 1):.1f}% | {mean_distance:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Episodes",
            "",
            "| # | obstacle | d | status | reason | distance | target | time |",
            "|---:|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {row['obstacle']} | {float(row['difficulty'] or 0.0):.2f} | "
            f"{row['status']} | {row['reason']} | {float(row['distance'] or 0.0):.2f} | "
            f"{row['target']} | {float(row['time'] or 0.0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "- **PASS** requires the obstacle-specific event plus a five-policy-step stable exit:",
            "  upright, forward velocity, support contact, no base/thigh/trunk terrain contact.",
            "- **FAIL** is a real capability failure: fall, illegal body contact, off-course, or timeout.",
            "- **ERROR** is a contract/runtime problem (checkpoint mismatch, missing asset, NaN), not a policy result.",
            "- Calf/shank contact is allowed during a climb, matching the training success criterion; only base,",
            "  trunk and thigh contact invalidate the episode.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python sim2sim.py eval --config configs/go2w.yaml --suite configs/suites/quick.yaml "
            "--checkpoint D:\\mujoco_test\\model_199.pt --out results\\quick",
            "```",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
