"""High-wall evaluator: every wall must be crossed inside its span."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEvaluator, EvaluatorContext
from .metrics import StepState
from .registry import EVALUATORS


class HighWallEvaluator(BaseEvaluator):
    kind = "high_wall"

    def __init__(self, ctx: EvaluatorContext) -> None:
        super().__init__(ctx)
        self.walls = sorted(self.build.features_of("wall"), key=lambda f: self.feature_along(f))
        self.wall_along = np.asarray([self.feature_along(f) for f in self.walls], dtype=np.float64)
        self.wall_half_length = np.asarray(
            [float(f.meta.get("half_length", 1.0)) for f in self.walls], dtype=np.float64
        )
        self.crossed = np.zeros(len(self.walls), dtype=bool)
        self._previous_along: float | None = None

    def _track_crossings(self, state: StepState) -> None:
        if self._previous_along is None:
            self._previous_along = float(state.along)
            return
        lateral = self.lateral_offset(state)
        for index, wall_along in enumerate(self.wall_along):
            if self.crossed[index]:
                continue
            if self._previous_along < wall_along <= float(state.along):
                # Crossing the plane outside the wall span means the robot
                # went around the wall; that must not count.
                if lateral <= float(self.wall_half_length[index]) + 0.20:
                    self.crossed[index] = True
        self._previous_along = float(state.along)

    def event_condition(self, state: StepState) -> bool:
        return bool(np.all(self.crossed)) and state.along >= self.target_distance

    def update(self, state: StepState) -> bool:
        self._track_crossings(state)
        return super().update(state)

    def details(self, state: StepState) -> dict[str, Any]:
        return {
            "walls_crossed": int(np.count_nonzero(self.crossed)),
            "walls_total": len(self.walls),
            "wall_heights": [round(float(f.meta.get("height", 0.0)), 5) for f in self.walls],
            "wall_along": [round(float(v), 5) for v in self.wall_along],
            "crossed": [bool(v) for v in self.crossed],
        }


EVALUATORS.register("high_wall", HighWallEvaluator)
EVALUATORS.register("wall", HighWallEvaluator)
