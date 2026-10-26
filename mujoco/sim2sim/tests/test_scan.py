from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from _helpers import ROOT

from dogsim.config import load_config
from dogsim.observation import HeightScanner
from dogsim.scene import compose_scene


class ScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "configs" / "go2w.yaml")
        cls.cfg.obstacle.type = "flat"
        cls.cfg.obstacle.difficulty = 0.0
        # Isolate the sign/geometry check from the actor's scan noise contract.
        cls.cfg.simulation.scan.noise_std = 0.0
        cls.cfg.simulation.scan.dropout_prob = 0.0
        cls.cfg.output.scene_xml = ROOT / "generated" / "test_scan.xml"
        cls.scene = compose_scene(cls.cfg, cls.cfg.output.scene_xml)
        cls.scanner = HeightScanner(cls.scene, cls.cfg)
        cls.scanner.reset(0)

    def _reset_pose(self) -> None:
        scene = self.scene
        spawn = scene.build.spawn
        scene.data.qpos[:7] = (spawn[0], spawn[1], spawn[2], 1.0, 0.0, 0.0, 0.0)
        scene.data.qpos[scene.joints.qpos_ids] = np.asarray(self.cfg.robot.default_angles)
        scene.data.qvel[:] = 0.0
        mujoco.mj_forward(scene.model, scene.data)

    def test_scan_is_positive_height(self) -> None:
        self._reset_pose()
        result = self.scanner.update()
        self.assertEqual(result.values.shape, (403,))
        self.assertGreater(float(result.values.min()), 0.0)
        self.assertLess(float(result.values.max()), 1.01)
        # The base is 0.46 m above the flat floor, so every ray should report
        # approximately that height.  A negative or near-zero mean is the
        # classic inverted-height-scan bug.
        self.assertAlmostEqual(float(result.values.mean()), 0.46, delta=0.02)
        self.assertAlmostEqual(result.valid_fraction, 1.0, places=6)

    def test_scan_holds_at_ten_hz(self) -> None:
        self._reset_pose()
        self.scanner.reset(0)
        flags = [self.scanner.update().refreshed for _ in range(7)]
        self.assertEqual(flags[0], True)
        self.assertEqual(flags[5], True)
        self.assertFalse(any(flags[1:5]))

    def test_scan_history_is_fixed_length(self) -> None:
        self._reset_pose()
        self.scanner.reset(0)
        self.scanner.update()
        history = self.scanner.scan_history(6)
        self.assertEqual(history.shape, (6, 403))


if __name__ == "__main__":
    unittest.main()
