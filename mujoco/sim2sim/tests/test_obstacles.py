from __future__ import annotations

import copy
import unittest

from _helpers import ROOT

from dogsim.config import load_config
from dogsim.scene import compose_scene


class ObstacleBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_config(ROOT / "configs" / "go2w.yaml")

    def test_every_obstacle_builds(self) -> None:
        for name in ("flat", "rough", "plum", "high_wall", "gap", "stairs", "low_bar", "course"):
            with self.subTest(obstacle=name):
                cfg = copy.deepcopy(self.base)
                cfg.obstacle.type = name
                cfg.obstacle.difficulty = 0.35
                cfg.output.scene_xml = ROOT / "generated" / f"test_{name}.xml"
                scene = compose_scene(cfg, cfg.output.scene_xml)
                self.assertGreater(scene.model.ngeom, 60)
                self.assertGreater(scene.model.nu, 0)
                self.assertGreater(len(scene.build.features), 0)
                self.assertGreater(scene.build.success_distance, 0.0)
                self.assertGreaterEqual(scene.build.spawn[2], scene.build.terrain_origin[2])
                self.assertGreater(len(scene.terrain_geom_ids), 0)
                self.assertGreater(len(scene.foot_geom_ids), 0)

    def test_difficulty_changes_geometry(self) -> None:
        cfg = copy.deepcopy(self.base)
        cfg.obstacle.type = "stairs"
        cfg.obstacle.difficulty = 0.0
        cfg.output.scene_xml = ROOT / "generated" / "test_stairs_easy.xml"
        easy = compose_scene(cfg, cfg.output.scene_xml)
        cfg.obstacle.difficulty = 1.0
        cfg.output.scene_xml = ROOT / "generated" / "test_stairs_hard.xml"
        hard = compose_scene(cfg, cfg.output.scene_xml)
        self.assertLess(easy.build.metadata["rise"], hard.build.metadata["rise"])


if __name__ == "__main__":
    unittest.main()
