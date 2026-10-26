"""Plum-blossom evaluator, aligned with the training foothold criterion."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEvaluator, EvaluatorContext
from .metrics import StepState
from .registry import EVALUATORS


class PlumEvaluator(BaseEvaluator):
    kind = "plum"

    def __init__(self, ctx: EvaluatorContext) -> None:
        super().__init__(ctx)
        self.stones = self.build.features_of("stone")
        self.visited = np.zeros(len(self.stones), dtype=bool)
        self.contact_steps = np.zeros(len(self.stones), dtype=np.int32)
        self.require_all = bool(self.evaluation.plum_require_all_stones)
        self.block_hold = int(self.evaluation.plum_block_hold_steps)
        self.top_tolerance = 0.055
        self.max_vertical_speed = 0.50
        self.wheel_radius = float(self.ctx.robot.wheel_radius)
        self.terrain_z = float(self.build.terrain_origin[2])
        self.route_done = False

    def _route_done(self) -> bool:
        if self.require_all:
            return bool(np.all(self.visited))
        # Two staggered rows of three.  A crossing follows one row; requiring
        # both rows forces a serpentine path the training task does not demand.
        row0 = bool(np.all(self.visited[:3])) if len(self.visited) >= 3 else False
        row1 = bool(np.all(self.visited[3:6])) if len(self.visited) >= 6 else False
        return row0 or row1

    def _update_stones(self, state: StepState) -> None:
        if not self.stones:
            return
        hits = np.zeros(len(self.stones), dtype=bool)
        for foot_index in range(state.foot_positions.shape[0]):
            if not bool(state.foot_contacts[foot_index]):
                continue
            foot = state.foot_positions[foot_index]
            foot_z = float(foot[2])
            vertical_speed = abs(float(state.foot_velocities[foot_index, 2]))
            if vertical_speed > self.max_vertical_speed:
                continue
            contact_z = foot_z - self.terrain_z - self.wheel_radius
            for stone_index, stone in enumerate(self.stones):
                half = stone.size[0] / 2.0, stone.size[1] / 2.0
                dx = abs(float(foot[0]) - float(stone.center[0]))
                dy = abs(float(foot[1]) - float(stone.center[1]))
                if dx > half[0] + 0.06 or dy > half[1] + 0.06:
                    continue
                top_rel = float(stone.top_z) - self.terrain_z
                if abs(contact_z - top_rel) > self.top_tolerance:
                    continue
                hits[stone_index] = True
        self.contact_steps = np.where(hits, self.contact_steps + 1, 0)
        self.visited |= self.contact_steps >= self.block_hold

    def event_condition(self, state: StepState) -> bool:
        self.route_done = self._route_done()
        return self.route_done and state.along >= self.target_distance

    def update(self, state: StepState) -> bool:
        self._update_stones(state)
        return super().update(state)

    def details(self, state: StepState) -> dict[str, Any]:
        rows = {
            "row_plus_y": [bool(v) for v in self.visited[:3]],
            "row_minus_y": [bool(v) for v in self.visited[3:6]],
        }
        return {
            "stones_visited": int(np.count_nonzero(self.visited)),
            "stones_total": len(self.stones),
            "stone_holds": [int(v) for v in self.contact_steps],
            "rows": rows,
            "route_done": bool(self.route_done),
            "require_all_stones": bool(self.require_all),
            "block_hold_steps": int(self.block_hold),
            "stone_top_z": [round(float(s.top_z), 5) for s in self.stones],
            "stone_size": [round(float(s.size[0]), 5) for s in self.stones],
        }


EVALUATORS.register("plum", PlumEvaluator)
EVALUATORS.register("plum_blossom", PlumEvaluator)
