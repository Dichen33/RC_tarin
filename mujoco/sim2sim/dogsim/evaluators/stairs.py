"""Stairs evaluator: reach the top platform and hold a stable pose there."""

from __future__ import annotations

import math
from typing import Any

from ..mathutils import local_to_world
from .base import BaseEvaluator, EvaluatorContext
from .metrics import StepState
from .registry import EVALUATORS


class StairsEvaluator(BaseEvaluator):
    kind = "stairs"

    def __init__(self, ctx: EvaluatorContext) -> None:
        super().__init__(ctx)
        top = self.build.feature("top_platform")
        origin = self.build.terrain_origin
        yaw = float(self.build.spawn[3])
        spawn = self.build.spawn
        x_start = float(top.meta["x_start"])
        x_end = float(top.meta["x_end"])
        self.top_z = float(top.top_z) - float(origin[2])
        self.top_start_along = (
            local_to_world(origin, yaw, x_start, 0.0, 0.0)[0] - spawn[0]
        ) * math.cos(yaw) + (
            local_to_world(origin, yaw, x_start, 0.0, 0.0)[1] - spawn[1]
        ) * math.sin(yaw)
        self.top_end_along = (
            local_to_world(origin, yaw, x_end, 0.0, 0.0)[0] - spawn[0]
        ) * math.cos(yaw) + (
            local_to_world(origin, yaw, x_end, 0.0, 0.0)[1] - spawn[1]
        ) * math.sin(yaw)
        # The meaningful exit is the top platform, not the far end of the
        # scene; report the distance to that platform.
        self.target_distance = float(self.top_start_along)
        self.reach_tolerance = float(self.evaluation.stair_reach_tolerance)
        self.reached_top = False
        self.max_height = 0.0

    def event_condition(self, state: StepState) -> bool:
        self.max_height = max(self.max_height, float(state.base_z_rel))
        if (
            state.along >= self.top_start_along - 0.05
            and state.base_z_rel >= self.top_z - self.reach_tolerance
        ):
            self.reached_top = True
        return self.reached_top

    def details(self, state: StepState) -> dict[str, Any]:
        return {
            "reached_top": bool(self.reached_top),
            "top_z": round(float(self.top_z), 5),
            "max_height": round(float(self.max_height), 5),
            "top_start_along": round(float(self.top_start_along), 5),
            "top_end_along": round(float(self.top_end_along), 5),
            "num_steps": int(self.build.metadata.get("num_steps", 0)),
            "rise": round(float(self.build.metadata.get("rise", 0.0)), 5),
            "tread": round(float(self.build.metadata.get("tread", 0.0)), 5),
        }


EVALUATORS.register("stairs", StairsEvaluator)
EVALUATORS.register("stair", StairsEvaluator)
