from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from _helpers import ROOT

from dogsim.config import load_config
from dogsim.panel import ControlPanel, SCENARIOS
from dogsim.policies import load_policy
from dogsim.scene import compose_scene
from dogsim.simulation import EpisodeRunner


class PanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "configs" / "course.yaml")
        cls.cfg.simulation.duration = 0.0
        cls.cfg.output.scene_xml = ROOT / "generated" / "test_panel.xml"
        cls.cfg.output.save_failed_trace = False
        cls.scene = compose_scene(cls.cfg, cls.cfg.output.scene_xml)
        cls.policy = load_policy(cls.cfg)
        cls.runner = EpisodeRunner(cls.scene, cls.cfg, cls.policy, seed=42)
        cls.runner.reset()
        cls.panel = ControlPanel(cls.runner, cls.scene)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.panel.close()

    def test_v2_map_has_obstacles(self) -> None:
        names = {item[0] for item in self.panel.map_geometries}
        self.assertTrue(any(name.startswith("high_wall") for name in names))
        self.assertTrue(any(name.startswith("stepping_stones") for name in names))
        self.assertTrue(any(name.startswith("platform") for name in names))
        xmin, xmax, ymin, ymax = self.panel.map_bounds
        self.assertLess(xmin, -6.0)
        self.assertGreater(xmax, 6.0)

    def test_scenario_buttons_and_keyboard(self) -> None:
        # course.yaml uses duration=0 (unlimited); it must not be interpreted
        # as "zero steps", which previously made the viewer look paused.
        self.assertIsNone(self.runner.max_steps)
        self.panel.select_scenario(2)  # plum platform
        self.assertAlmostEqual(self.runner.spawn[0], 3.225)
        self.assertAlmostEqual(self.runner.spawn[2], 0.65)
        self.panel.key("w")
        self.assertAlmostEqual(float(self.panel.command[0]), 0.25)
        self.panel.start_scenario_navigation()
        self.assertTrue(self.panel.navigation.active)
        self.panel.key("x")
        self.assertFalse(self.panel.navigation.active)
        self.assertTrue(all(abs(float(v)) < 1e-9 for v in self.panel.command))
        self.panel.draw()

    def test_all_scenarios_are_reachable(self) -> None:
        for index, scenario in enumerate(SCENARIOS):
            self.panel.select_scenario(index)
            self.assertAlmostEqual(self.runner.spawn[0], scenario.spawn[0])
            self.assertAlmostEqual(self.runner.spawn[1], scenario.spawn[1])

    def test_model_picker_page(self) -> None:
        from dogsim.model_catalog import ModelCatalog

        model_dir = ROOT / "generated" / "test_model_catalog"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_1234.pt").write_bytes(b"fake checkpoint")
        catalog = ModelCatalog(roots=[model_dir])
        entries = catalog.scan()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].iteration, 1234)
        self.panel.model_catalog = catalog
        self.panel.model_loader = lambda entry, scale: (True, f"fake load {entry.name}")
        self.panel.open_model_page()
        self.assertTrue(self.panel.model_page)
        self.assertGreater(len(self.panel.model_entries), 0)
        self.panel._set_model_selection(0)
        self.panel.load_selected_model()
        self.assertFalse(self.panel.model_page)
        self.assertIsNotNone(self.panel.current_model)
        self.panel.draw()


if __name__ == "__main__":
    unittest.main()
