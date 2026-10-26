"""One reusable stepping runtime for every obstacle and policy."""

from __future__ import annotations

import csv
import json
import math
import time as wall_time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .config import RunConfig
from .evaluators import EvaluatorContext, MetricsTracker, ObstacleResult, make_evaluator
from .observation import HeightScanner, ObservationPipeline
from .policies import PolicyBackend, load_policy
from .randomization import ControlParams, Randomizer
from .scene import Scene, compose_scene


@dataclass
class EpisodeResult:
    config: dict[str, Any]
    result: ObstacleResult | None
    status: str
    error: str | None
    steps: int
    time: float
    seed: int
    scene_xml: str = ""
    trace_path: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "status": self.status,
            "error": self.error,
            "steps": int(self.steps),
            "time": round(float(self.time), 5),
            "seed": int(self.seed),
            "scene_xml": self.scene_xml,
            "trace_path": self.trace_path,
            "result": None if self.result is None else self.result.as_dict(),
            "extras": self.extras,
        }


class EpisodeRunner:
    def __init__(
        self,
        scene: Scene,
        cfg: RunConfig,
        policy: PolicyBackend,
        seed: int | None = None,
    ) -> None:
        self.scene = scene
        self.cfg = cfg
        self.policy = policy
        self.seed = int(cfg.simulation.randomization.seed if seed is None else seed)
        self.command = np.clip(
            np.asarray(cfg.command, dtype=np.float32),
            -np.asarray(cfg.max_command, dtype=np.float32),
            np.asarray(cfg.max_command, dtype=np.float32),
        )
        self.scanner = HeightScanner(scene, cfg)
        self.observation = ObservationPipeline(scene, cfg)
        self.tracker = MetricsTracker(scene, cfg)
        self.randomizer = Randomizer(scene, cfg)
        self.ctx = EvaluatorContext(
            scene=scene,
            cfg=cfg,
            build=scene.build,
            evaluation=cfg.evaluation,
        )
        self.evaluator = make_evaluator(cfg.evaluation.kind, self.ctx)
        self.spawn = (
            tuple(float(v) for v in cfg.spawn_override)
            if cfg.spawn_override is not None
            else tuple(float(v) for v in scene.build.spawn)
        )
        self.duration = float(cfg.evaluation.duration or cfg.simulation.duration)
        # duration 0 means "unlimited" for the interactive viewer.  A numeric
        # max_steps of 0 would otherwise terminate the episode after one step
        # and make the UI look permanently paused.
        self.max_steps = (
            None
            if self.duration <= 0.0
            else int(round(self.duration / cfg.simulation.control_dt))
        )
        self.control_dt = cfg.simulation.control_dt
        self.physics_dt = cfg.simulation.physics_dt
        self.default_angles = np.asarray(cfg.robot.default_angles, dtype=np.float32)
        self.position_default = self.default_angles[scene.joints.position_action_indices]
        self.position_target = self.position_default.copy()
        self.velocity_target = np.zeros(len(scene.joints.velocity_action_indices), dtype=np.float32)
        self.executed_action = np.zeros(cfg.robot.action_dim, dtype=np.float32)
        self.action_queue: list[np.ndarray] = []
        self.control = ControlParams(
            motor_response=1.0,
            action_delay_steps=0,
            control_steps=cfg.simulation.control_decimation,
        )
        self.step_index = 0
        self.time = 0.0
        self.trace: list[dict[str, Any]] = []
        self._trace_counter = 0
        self.last_state = None

    def set_command(self, command: np.ndarray) -> None:
        self.command = np.clip(
            np.asarray(command, dtype=np.float32),
            -np.asarray(self.cfg.max_command, dtype=np.float32),
            np.asarray(self.cfg.max_command, dtype=np.float32),
        )
        self.observation.set_command(self.command)

    def set_spawn(self, spawn: tuple[float, float, float, float] | list[float]) -> None:
        """Move the next reset to a UI-selected obstacle start."""

        if len(spawn) != 4:
            raise ValueError("spawn must contain x y z yaw")
        self.spawn = tuple(float(v) for v in spawn)

    def set_policy(self, policy: PolicyBackend) -> None:
        """Swap the inference backend for the UI model picker."""

        self.policy = policy
        self.policy.reset()

    @property
    def collision_count(self) -> int:
        return len(self.tracker._pair_set)

    @property
    def fallen(self) -> bool:
        return bool(self.last_state.fallen) if self.last_state is not None else False

    @property
    def elapsed(self) -> float:
        return float(self.time)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        cfg = self.cfg
        self.control = self.randomizer.reset(self.seed)
        mujoco.mj_resetData(self.scene.model, self.scene.data)
        data = self.scene.data
        x, y, z, yaw = self.spawn
        data.qpos[:7] = (x, y, z, math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        data.qpos[self.scene.joints.qpos_ids] = self.default_angles.astype(np.float64)
        data.qvel[:] = 0.0
        data.ctrl[self.scene.joints.position_actuator_ids] = self.position_default
        if self.scene.joints.velocity_actuator_ids.size:
            data.ctrl[self.scene.joints.velocity_actuator_ids] = 0.0
        mujoco.mj_forward(self.scene.model, data)

        self.policy.reset()
        self.scanner.reset(self.seed)
        self.observation.reset(self.seed)
        self.observation.set_command(self.command)
        self.observation.set_executed_action(np.zeros(cfg.robot.action_dim, dtype=np.float32))
        self.observation.update()
        self.scanner.update()
        self.tracker.reset()
        self.evaluator = make_evaluator(cfg.evaluation.kind, self.ctx)
        self.evaluator.setup()
        self.position_target = self.position_default.copy()
        self.velocity_target = np.zeros(len(self.scene.joints.velocity_action_indices), dtype=np.float32)
        self.executed_action = np.zeros(cfg.robot.action_dim, dtype=np.float32)
        self.action_queue = [
            np.zeros(cfg.robot.action_dim, dtype=np.float32)
            for _ in range(int(self.control.action_delay_steps))
        ]
        self.step_index = 0
        self.time = 0.0
        self.trace.clear()
        self._trace_counter = 0
        self.last_state = None

    # ------------------------------------------------------------------
    def _apply_action(self, action: np.ndarray) -> None:
        data = self.scene.data
        position_scale = self.cfg.robot.position_action_scale
        velocity_scale = self.cfg.robot.velocity_action_scale
        position_indices = self.scene.joints.position_action_indices
        velocity_indices = self.scene.joints.velocity_action_indices
        position_target = self.position_default + action[position_indices] * position_scale
        self.position_target += self.control.motor_response * (position_target - self.position_target)
        data.ctrl[self.scene.joints.position_actuator_ids] = self.position_target
        if velocity_indices.size:
            velocity_target = action[velocity_indices] * velocity_scale
            self.velocity_target += self.control.motor_response * (velocity_target - self.velocity_target)
            data.ctrl[self.scene.joints.velocity_actuator_ids] = self.velocity_target

    # ------------------------------------------------------------------
    def step(self) -> bool:
        scan = self.scanner.update()
        self.observation.update()
        proprio = self.observation.term_major_history()
        scan_history = self.scanner.scan_history(self.policy.contract.history_length)
        action = self.policy.act(proprio, scan.values, scan_history)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("Policy produced a non-finite action")
        action = np.clip(action, -5.0, 5.0).astype(np.float32)
        if self.control.action_delay_steps:
            self.action_queue.append(action.copy())
            executed = self.action_queue.pop(0)
        else:
            executed = action
        self.executed_action = executed.copy()
        self.observation.set_executed_action(self.executed_action)
        self._apply_action(self.executed_action)

        for _ in range(int(self.control.control_steps)):
            mujoco.mj_step(self.scene.model, self.scene.data)
        if not np.all(np.isfinite(self.scene.data.qpos)) or not np.all(np.isfinite(self.scene.data.qvel)):
            raise FloatingPointError("Physics state became non-finite")

        self.time += self.control.control_steps * self.physics_dt
        self.observation.advance(self.control_dt)
        self.step_index += 1
        state = self.tracker.update(self.step_index, self.time)
        self.last_state = state
        finished = self.evaluator.update(state)

        if self.step_index % max(1, self.cfg.output.trace_every) == 0:
            record = state.as_dict()
            record.update(
                {
                    "action_norm": round(float(np.linalg.norm(self.executed_action)), 5),
                    "scan_valid": round(float(scan.valid_fraction), 5),
                    "scan_refreshed": bool(scan.refreshed),
                    "control_steps": int(self.control.control_steps),
                    "motor_response": round(float(self.control.motor_response), 5),
                }
            )
            self.trace.append(record)

        if state.fallen and self.cfg.evaluation.stop_on_fall:
            return True
        if finished:
            return True
        if self.max_steps is not None and self.step_index >= self.max_steps:
            return True
        return False

    # ------------------------------------------------------------------
    def run(self, verbose: bool = False) -> ObstacleResult:
        self.reset()
        wall_start = wall_time.perf_counter()
        error: str | None = None
        try:
            while True:
                if self.step():
                    break
                if verbose and self.step_index % 250 == 0 and self.last_state is not None:
                    state = self.last_state
                    print(
                        f"[Run] t={self.time:5.2f}s along={state.along:6.2f} "
                        f"z={state.base_z_rel:.3f} tilt={state.tilt:.2f} "
                        f"vx={state.forward_velocity:5.2f} "
                        f"support={state.support_contact} illegal={state.illegal_contact}"
                    )
        except Exception as exc:  # noqa: BLE001 - reported as an episode error
            error = f"{type(exc).__name__}: {exc}"
        elapsed = wall_time.perf_counter() - wall_start
        result = self.evaluator.result(self.time, self.step_index)
        if error is not None:
            result.status = "ERROR"
            result.reason = error
        result.metrics["wall_time_s"] = round(elapsed, 4)
        return result


# ---------------------------------------------------------------------------
# High-level helper used by the CLI and the suite
# ---------------------------------------------------------------------------


def run_episode(
    cfg: RunConfig,
    policy: PolicyBackend | None = None,
    seed: int | None = None,
    scene_xml: Path | None = None,
    verbose: bool = False,
    save_trace: bool | None = None,
) -> EpisodeResult:
    scene_xml = Path(scene_xml) if scene_xml is not None else cfg.output.scene_xml
    result: EpisodeResult
    try:
        policy = policy or load_policy(cfg)
        # The checkpoint is authoritative for history and scan geometry.
        cfg.policy.history_length = policy.contract.history_length
        cfg.policy.scan_dim = policy.contract.scan_dim
        cfg.policy.scan_size = policy.contract.scan_size
        cfg.validate()
        scene = compose_scene(cfg, scene_xml)
        runner = EpisodeRunner(scene, cfg, policy, seed=seed)
        obstacle_result = runner.run(verbose=verbose)
        trace_path = None
        should_save = cfg.output.save_trace if save_trace is None else bool(save_trace)
        if should_save or (cfg.output.save_failed_trace and not obstacle_result.passed):
            trace_path = _save_trace(cfg.output.results_dir, cfg, runner, obstacle_result)
        return EpisodeResult(
            config=cfg.as_serializable(),
            result=obstacle_result,
            status=obstacle_result.status,
            error=None if obstacle_result.status != "ERROR" else obstacle_result.reason,
            steps=runner.step_index,
            time=runner.time,
            seed=int(cfg.simulation.randomization.seed if seed is None else seed),
            scene_xml=str(scene_xml),
            trace_path=None if trace_path is None else str(trace_path),
        )
    except Exception as exc:  # noqa: BLE001 - a bad checkpoint must not kill the suite
        message = f"{type(exc).__name__}: {exc}"
        obstacle_result = ObstacleResult(
            kind=cfg.obstacle.type,
            status="ERROR",
            reason=message,
            distance=0.0,
            duration=0.0,
            steps=0,
            stable_steps=0,
        )
        return EpisodeResult(
            config=cfg.as_serializable(),
            result=obstacle_result,
            status="ERROR",
            error=message,
            steps=0,
            time=0.0,
            seed=int(cfg.simulation.randomization.seed if seed is None else seed),
            scene_xml=str(scene_xml),
        )


def _save_trace(results_dir: Path, cfg: RunConfig, runner: EpisodeRunner, result: ObstacleResult) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"trace_{cfg.obstacle.type}_d{cfg.obstacle.difficulty:.2f}_"
        f"{result.status.lower()}_{cfg.simulation.randomization.seed}"
    )
    path = results_dir / f"{stem}.json"
    payload = {
        "config": cfg.as_serializable(),
        "result": result.as_dict(),
        "trace": runner.trace,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    csv_path = path.with_suffix(".csv")
    if runner.trace:
        keys = sorted(runner.trace[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for record in runner.trace:
                writer.writerow(record)
    return path
