from __future__ import annotations

import unittest

import numpy as np

from _helpers import ROOT, build_gap, build_wall, make_state

from dogsim.config import load_config
from dogsim.evaluators import EvaluatorContext
from dogsim.evaluators.gap import GapEvaluator
from dogsim.evaluators.high_wall import HighWallEvaluator
from dogsim.evaluators.plum import PlumEvaluator
from dogsim.obstacles import Feature, ObstacleBuild


class EvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "configs" / "go2w.yaml")
        cls.cfg.obstacle.type = "flat"
        cls.cfg.obstacle.difficulty = 0.0

    def _ctx(self, build: ObstacleBuild) -> EvaluatorContext:
        return EvaluatorContext(scene=None, cfg=self.cfg, build=build, evaluation=self.cfg.evaluation)  # type: ignore[arg-type]

    def test_gap_pass(self) -> None:
        evaluator = GapEvaluator(self._ctx(build_gap()))
        sequence = [1.8, 2.2, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65]
        for index, along in enumerate(sequence):
            done = evaluator.update(make_state(along, step=index + 1, time=0.02 * (index + 1)))
        result = evaluator.result(1.0, len(sequence))
        self.assertEqual(result.status, "PASS", result.reason)
        self.assertTrue(result.metrics["gap_crossed"])

    def test_wall_pass(self) -> None:
        evaluator = HighWallEvaluator(self._ctx(build_wall()))
        sequence = [1.0, 1.5, 3.0, 4.6, 4.9, 4.95, 5.0, 5.05, 5.1, 5.15]
        for index, along in enumerate(sequence):
            evaluator.update(make_state(along, step=index + 1, time=0.02 * (index + 1)))
        result = evaluator.result(1.0, len(sequence))
        self.assertEqual(result.status, "PASS", result.reason)
        self.assertEqual(result.metrics["walls_crossed"], 3)

    def test_wall_does_not_count_going_around(self) -> None:
        evaluator = HighWallEvaluator(self._ctx(build_wall()))
        # Lateral offset > wall half length means the robot went around the
        # wall.  The plane crossing must not count as a wall clearance.
        sequence = [1.0, 1.5, 3.0, 4.6, 4.9]
        for index, along in enumerate(sequence):
            evaluator.update(
                make_state(along, lateral=1.2, step=index + 1, time=0.02 * (index + 1))
            )
        result = evaluator.result(1.0, len(sequence))
        self.assertLess(result.metrics["walls_crossed"], 3)

    def test_plum_row_pass(self) -> None:
        terrain_z = 0.189
        blocks = []
        for index, (x, y) in enumerate(((0.6, 0.2), (1.2, 0.2), (1.8, 0.2), (0.9, -0.2), (1.5, -0.2), (2.1, -0.2))):
            blocks.append(
                Feature(
                    name=f"block_{index}",
                    kind="stone",
                    center=(x, y, 0.177),
                    size=(0.204, 0.204, 0.377),
                    top_z=0.177,
                    meta={"row": 0 if y > 0 else 1},
                )
            )
        build = ObstacleBuild(
            type="plum",
            spawn=(0.0, 0.0, 0.639, 0.0),
            terrain_origin=(0.0, 0.0, terrain_z),
            success_distance=2.7,
            features=blocks,
            lane_width=2.4,
        )
        evaluator = PlumEvaluator(self._ctx(build))
        foot_positions = np.zeros((4, 3), dtype=np.float64)
        foot_contacts = np.asarray((True, False, False, False), dtype=bool)
        foot_velocities = np.zeros((4, 3), dtype=np.float64)

        def feed_stone(x: float, y: float, step: int) -> None:
            foot_positions[0] = (x, y, 0.177 + 0.086)
            evaluator.update(
                make_state(
                    0.7,
                    z=0.30,
                    step=step,
                    time=0.02 * step,
                    foot_positions=foot_positions,
                    foot_contacts=foot_contacts,
                    foot_velocities=foot_velocities,
                )
            )

        step = 0
        for x, y in ((0.6, 0.2), (1.2, 0.2), (1.8, 0.2)):
            for _ in range(3):
                step += 1
                feed_stone(x, y, step)
        for _ in range(6):
            step += 1
            evaluator.update(make_state(2.8, z=0.30, step=step, time=0.02 * step))
        result = evaluator.result(2.0, step)
        self.assertEqual(result.status, "PASS", result.reason)
        self.assertTrue(result.metrics["route_done"])
        self.assertEqual(result.metrics["stones_visited"], 3)


if __name__ == "__main__":
    unittest.main()
