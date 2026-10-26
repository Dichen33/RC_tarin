from __future__ import annotations

import copy
import unittest

from _helpers import ROOT

from dogsim.config import load_config
from dogsim.simulation import run_episode


class FlatEpisodeTest(unittest.TestCase):
    def test_checkpoint_drives_on_flat(self) -> None:
        cfg = load_config(ROOT / "configs" / "go2w.yaml")
        cfg.obstacle.type = "flat"
        cfg.obstacle.difficulty = 0.0
        cfg.simulation.duration = 15.0
        cfg.output.scene_xml = ROOT / "generated" / "test_flat_episode.xml"
        cfg.output.save_trace = False
        cfg.output.save_failed_trace = False
        result = run_episode(cfg, seed=42)
        self.assertIsNotNone(result.result)
        self.assertEqual(result.status, "PASS", result.error or result.result.reason)
        assert result.result is not None
        self.assertGreaterEqual(result.result.distance, 4.0)


if __name__ == "__main__":
    unittest.main()
