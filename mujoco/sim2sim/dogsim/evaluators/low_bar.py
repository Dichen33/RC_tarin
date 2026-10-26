"""Low-bar evaluator (optional clearance test)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEvaluator, EvaluatorContext
from .metrics import StepState
from .registry import EVALUATORS


class LowBarEvaluator(BaseEvaluator):
    kind = "low_bar"

    def __init__(self, ctx: EvaluatorContext) -> None:
        super().__init__(ctx)
        self.bars = sorted(self.build.features_of("bar"), key=lambda f: self.feature_along(f))
        self.bar_along = np.asarray([self.feature_along(f) for f in self.bars], dtype=np.float64)
        self.crossed = np.zeros(len(self.bars), dtype=bool)
        self._previous_along: float | None = None

    def event_condition(self, state: StepState) -> bool:
        if self._previous_along is None:
            self._previous_along = float(state.along)
        for index, along in enumerate(self.bar_along):
            if not self.crossed[index] and self._previous_along < along <= float(state.along):
                self.crossed[index] = True
        self._previous_along = float(state.along)
        return bool(np.all(self.crossed)) and state.along >= self.target_distance

    def details(self, state: StepState) -> dict[str, Any]:
        return {
            "bars_crossed": int(np.count_nonzero(self.crossed)),
            "bars_total": len(self.bars),
            "clearance": round(float(self.build.metadata.get("clearance", 0.0)), 5),
        }


EVALUATORS.register("low_bar", LowBarEvaluator)
