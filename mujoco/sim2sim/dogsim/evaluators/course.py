"""Generic integrated-course evaluator (progress + stable exit)."""

from __future__ import annotations

from typing import Any

from .base import BaseEvaluator
from .metrics import StepState
from .registry import EVALUATORS


class CourseEvaluator(BaseEvaluator):
    kind = "course"

    def details(self, state: StepState) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for feature in self.build.features:
            counts[feature.kind] = counts.get(feature.kind, 0) + 1
        return {
            "course_geoms": counts,
            "course_xml": self.build.metadata.get("course_xml", ""),
        }


EVALUATORS.register("course", CourseEvaluator)
