from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dogsim.evaluators import EvaluatorContext
from dogsim.evaluators.metrics import StepState
from dogsim.obstacles import Feature, ObstacleBuild


def make_state(
    along: float,
    *,
    z: float = 0.30,
    tilt: float = 0.10,
    vx: float = 0.30,
    vz: float = 0.0,
    support: bool = True,
    illegal: bool = False,
    fallen: bool = False,
    foot_positions: np.ndarray | None = None,
    foot_contacts: np.ndarray | None = None,
    foot_velocities: np.ndarray | None = None,
    lateral: float = 0.0,
    step: int = 1,
    time: float = 0.02,
) -> StepState:
    if foot_positions is None:
        foot_positions = np.zeros((4, 3), dtype=np.float64)
    if foot_contacts is None:
        foot_contacts = np.asarray((True, True, True, True), dtype=bool)
    if foot_velocities is None:
        foot_velocities = np.zeros((4, 3), dtype=np.float64)
    return StepState(
        step=step,
        time=time,
        base_pos=np.asarray((along, lateral, z), dtype=np.float64),
        base_quat=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        base_velocity=np.asarray((vx, 0.0, vz), dtype=np.float64),
        base_z_rel=z,
        tilt=tilt,
        forward_velocity=vx,
        vertical_velocity=abs(vz),
        support_contact=support,
        illegal_contact=illegal,
        base_contact=False,
        thigh_contact=False,
        calf_contact=False,
        foot_contacts=np.asarray(foot_contacts, dtype=bool),
        foot_positions=np.asarray(foot_positions, dtype=np.float64),
        foot_velocities=np.asarray(foot_velocities, dtype=np.float64),
        collision_pairs=0,
        fallen=fallen,
        along=float(along),
    )


def build_gap() -> ObstacleBuild:
    gap = Feature(
        name="gap",
        kind="gap",
        center=(1.5, 0.0, -0.1),
        size=(0.89, 2.4, 0.2),
        top_z=0.0,
        meta={"start_x": 1.15, "end_x": 2.04, "width": 0.89, "depth": 0.2},
    )
    return ObstacleBuild(
        type="gap",
        spawn=(0.0, 0.0, 0.45, 0.0),
        terrain_origin=(0.0, 0.0, 0.0),
        success_distance=3.35,
        features=[gap],
        lane_width=2.4,
    )


def build_wall() -> ObstacleBuild:
    walls = [
        Feature(
            name=f"wall_{i}",
            kind="wall",
            center=(x, 0.0, 0.2),
            size=(0.05, 1.8, 0.4),
            top_z=0.2,
            meta={"x": x, "half_length": 0.9, "height": 0.2},
        )
        for i, x in enumerate((1.4, 2.95, 4.5))
    ]
    return ObstacleBuild(
        type="high_wall",
        spawn=(0.0, 0.0, 0.45, 0.0),
        terrain_origin=(0.0, 0.0, 0.0),
        success_distance=4.8,
        features=walls,
        lane_width=2.4,
    )
