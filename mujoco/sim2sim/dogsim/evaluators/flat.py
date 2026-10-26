"""Flat-ground evaluator: distance plus a stable exit hold."""

from __future__ import annotations

from .base import BaseEvaluator
from .registry import EVALUATORS


class FlatEvaluator(BaseEvaluator):
    kind = "flat"


EVALUATORS.register("flat", FlatEvaluator)
