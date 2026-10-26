"""Evaluator contract and common success bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import EvaluationConfig, RunConfig
from ..obstacles import ObstacleBuild
from ..scene import Scene
from .metrics import MetricsTracker, StepState


@dataclass
class EvaluatorContext:
    scene: Scene
    cfg: RunConfig
    build: ObstacleBuild
    evaluation: EvaluationConfig

    @property
    def robot(self):
        return self.cfg.robot


@dataclass
class ObstacleResult:
    kind: str
    status: str
    reason: str
    distance: float
    duration: float
    steps: int
    stable_steps: int
    metrics: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "reason": self.reason,
            "distance": round(float(self.distance), 5),
            "duration": round(float(self.duration), 5),
            "steps": int(self.steps),
            "stable_steps": int(self.stable_steps),
            "metrics": self.metrics,
            "features": self.features,
        }


class BaseEvaluator:
    kind = "base"

    def __init__(self, ctx: EvaluatorContext) -> None:
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.evaluation = ctx.evaluation
        self.build = ctx.build
        self.target_distance = (
            float(self.evaluation.success_distance)
            if self.evaluation.success_distance is not None
            else float(self.build.success_distance)
        )
        self.required_hold = int(self.evaluation.stable_steps)
        self.stable_steps = 0
        self.hold_steps = 0
        self.max_along = 0.0
        self.max_lateral = 0.0
        self.max_tilt_seen = 0.0
        self.min_height_seen = float("inf")
        self.illegal_seen = False
        self.base_contact_seen = False
        self.off_course = False
        self.done = False
        self.passed = False
        self.reason = "running"
        self.last_state: StepState | None = None
        lane_limit = self.build.lane_width / 2.0 + 0.75
        if self.evaluation.max_lateral_deviation is not None:
            lane_limit = float(self.evaluation.max_lateral_deviation)
        self.lateral_limit = lane_limit

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------
    def event_condition(self, state: StepState) -> bool:
        """The obstacle-specific event that must be followed by a stable hold."""

        return state.along >= self.target_distance

    def details(self, state: StepState) -> dict[str, Any]:
        return {}

    def setup(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Common logic
    # ------------------------------------------------------------------
    def lateral_offset(self, state: StepState) -> float:
        dx = float(state.base_pos[0] - self.build.spawn[0])
        dy = float(state.base_pos[1] - self.build.spawn[1])
        yaw = float(self.build.spawn[3])
        return abs(-np.sin(yaw) * dx + np.cos(yaw) * dy)

    def feature_along(self, feature) -> float:
        dx = float(feature.center[0] - self.build.spawn[0])
        dy = float(feature.center[1] - self.build.spawn[1])
        yaw = float(self.build.spawn[3])
        return dx * np.cos(yaw) + dy * np.sin(yaw)

    def stable_candidate(self, state: StepState) -> bool:
        if state.fallen or state.illegal_contact:
            return False
        if state.tilt >= self.evaluation.max_tilt:
            return False
        if state.base_z_rel <= self.evaluation.min_height:
            return False
        if state.forward_velocity <= self.evaluation.min_forward_velocity:
            return False
        if state.vertical_velocity >= self.evaluation.max_vertical_velocity:
            return False
        if self.evaluation.require_support and not state.support_contact:
            return False
        if float(self.cfg.command[0]) <= self.evaluation.command_threshold:
            return False
        return True

    def update(self, state: StepState) -> bool:
        """Update bookkeeping and return True when the episode has finished."""

        self.last_state = state
        self.max_along = max(self.max_along, float(state.along))
        self.max_lateral = max(self.max_lateral, self.lateral_offset(state))
        self.max_tilt_seen = max(self.max_tilt_seen, float(state.tilt))
        self.min_height_seen = min(self.min_height_seen, float(state.base_z_rel))
        self.illegal_seen = self.illegal_seen or bool(state.illegal_contact)
        self.base_contact_seen = self.base_contact_seen or bool(state.base_contact)
        if self.max_lateral > self.lateral_limit:
            self.off_course = True

        stable = self.stable_candidate(state)
        if stable:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        event = bool(self.event_condition(state))
        if event and stable:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        if self.hold_steps >= self.required_hold:
            self.done = True
            self.passed = True
            self.reason = "success"
            return True

        if state.fallen:
            self.reason = "fallen"
            if self.evaluation.stop_on_fall:
                self.done = True
                return True
        if self.illegal_seen:
            self.reason = "illegal_contact"
        elif self.off_course:
            self.reason = "off_course"
        elif event and self.hold_steps < self.required_hold:
            self.reason = f"exit_not_stable({self.hold_steps}/{self.required_hold})"
        else:
            self.reason = "running"
        return False

    def result(self, elapsed: float, steps: int) -> ObstacleResult:
        state = self.last_state
        distance = float(state.along) if state is not None else 0.0
        if self.passed:
            status = "PASS"
            reason = "success"
        elif self.illegal_seen:
            status = "FAIL"
            reason = "illegal_contact"
        elif self.off_course:
            status = "FAIL"
            reason = "off_course"
        elif state is not None and state.fallen:
            status = "FAIL"
            reason = "fallen"
        else:
            status = "FAIL"
            reason = self.reason if self.reason != "running" else "timeout"
        metrics = {
            "final_along": round(distance, 5),
            "max_along": round(self.max_along, 5),
            "target_distance": round(self.target_distance, 5),
            "stable_steps": int(self.stable_steps),
            "hold_steps": int(self.hold_steps),
            "max_lateral": round(self.max_lateral, 5),
            "illegal_contact": bool(self.illegal_seen),
            "base_contact": bool(self.base_contact_seen),
            "off_course": bool(self.off_course),
            "final_z_rel": round(float(state.base_z_rel), 5) if state is not None else 0.0,
            "max_tilt": round(float(self.max_tilt_seen), 5),
            "min_z_rel": round(float(self.min_height_seen), 5),
        }
        if state is not None:
            metrics.update(self.details(state))
        return ObstacleResult(
            kind=self.kind,
            status=status,
            reason=reason,
            distance=distance,
            duration=elapsed,
            steps=steps,
            stable_steps=self.stable_steps,
            metrics=metrics,
            features=self.build.summary(),
        )


class TrackerMixin:
    """Convenience for evaluators that need the shared metrics tracker."""

    tracker: MetricsTracker
