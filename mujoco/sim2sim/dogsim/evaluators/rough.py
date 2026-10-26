"""Rough-ground evaluator."""

from __future__ import annotations

from typing import Any

from .base import BaseEvaluator
from .metrics import StepState
from .registry import EVALUATORS


class RoughEvaluator(BaseEvaluator):
    kind = "rough"

    def details(self, state: StepState) -> dict[str, Any]:
        return {
            "rough_max_height": self.build.metadata.get("max_height", 0.0),
            "rough_cell": self.build.metadata.get("cell", 0.0),
        }


EVALUATORS.register("rough", RoughEvaluator)
