"""Shared evaluator registry."""

from __future__ import annotations

from ..registry import Registry
from .base import BaseEvaluator

EVALUATORS: Registry[type[BaseEvaluator]] = Registry("evaluator")
