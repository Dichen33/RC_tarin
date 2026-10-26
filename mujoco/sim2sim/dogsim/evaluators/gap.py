"""Gap evaluator: enter, cross the far edge, then hold a stable exit."""

from __future__ import annotations

import math
from typing import Any

from ..mathutils import local_to_world
from .base import BaseEvaluator, EvaluatorContext
from .metrics import StepState
from .registry import EVALUATORS


class GapEvaluator(BaseEvaluator):
    kind = "gap"

    def __init__(self, ctx: EvaluatorContext) -> None:
        super().__init__(ctx)
        gap = self.build.feature("gap")
        origin = self.build.terrain_origin
        yaw = float(self.build.spawn[3])
        spawn = self.build.spawn

        def along(local_x: float) -> float:
            point = local_to_world(origin, yaw, local_x, 0.0, 0.0)
            return (point[0] - spawn[0]) * math.cos(yaw) + (point[1] - spawn[1]) * math.sin(yaw)

        self.gap_start = along(float(gap.meta["start_x"]))
        self.gap_end = along(float(gap.meta["end_x"]))
        self.width = float(gap.meta["width"])
        self.depth = float(gap.meta["depth"])
        self.entered = False
        self.crossed = False

    def event_condition(self, state: StepState) -> bool:
        if state.along >= self.gap_start:
            self.entered = True
        if state.along >= self.gap_end:
            self.crossed = True
        return self.crossed and state.along >= self.target_distance

    def details(self, state: StepState) -> dict[str, Any]:
        return {
            "gap_entered": bool(self.entered),
            "gap_crossed": bool(self.crossed),
            "gap_width": round(self.width, 5),
            "gap_depth": round(self.depth, 5),
            "gap_start_along": round(self.gap_start, 5),
            "gap_end_along": round(self.gap_end, 5),
        }


EVALUATORS.register("gap", GapEvaluator)
EVALUATORS.register("ditch", GapEvaluator)
