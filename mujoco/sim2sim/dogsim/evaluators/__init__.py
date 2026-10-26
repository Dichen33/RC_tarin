"""Evaluator bricks and the factory used by the runtime/suite."""

from __future__ import annotations

from .base import BaseEvaluator, EvaluatorContext, ObstacleResult
from .metrics import MetricsTracker, StepState
from .registry import EVALUATORS

# Registration side effects.
from .flat import FlatEvaluator  # noqa: F401,E402
from .rough import RoughEvaluator  # noqa: F401,E402
from .plum import PlumEvaluator  # noqa: F401,E402
from .high_wall import HighWallEvaluator  # noqa: F401,E402
from .gap import GapEvaluator  # noqa: F401,E402
from .stairs import StairsEvaluator  # noqa: F401,E402
from .low_bar import LowBarEvaluator  # noqa: F401,E402
from .course import CourseEvaluator  # noqa: F401,E402


class ProgressEvaluator(BaseEvaluator):
    """Generic distance + stable-exit evaluator for future obstacle bricks."""

    kind = "progress"


def make_evaluator(kind: str, ctx: EvaluatorContext) -> BaseEvaluator:
    requested = (kind or "auto").strip().lower()
    if requested == "auto":
        requested = ctx.build.type
    if requested in EVALUATORS:
        return EVALUATORS.get(requested)(ctx)
    # A new obstacle can be evaluated before a dedicated evaluator exists;
    # progress is explicit rather than silently reporting PASS.
    evaluator = ProgressEvaluator(ctx)
    evaluator.kind = requested
    return evaluator


def evaluator_names() -> tuple[str, ...]:
    return EVALUATORS.names()


__all__ = [
    "BaseEvaluator",
    "EvaluatorContext",
    "ObstacleResult",
    "MetricsTracker",
    "StepState",
    "EVALUATORS",
    "make_evaluator",
    "evaluator_names",
    "FlatEvaluator",
    "RoughEvaluator",
    "PlumEvaluator",
    "HighWallEvaluator",
    "GapEvaluator",
    "StairsEvaluator",
    "LowBarEvaluator",
    "CourseEvaluator",
    "ProgressEvaluator",
]
