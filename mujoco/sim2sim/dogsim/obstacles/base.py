"""Obstacle module contract.

An obstacle module is a self-contained brick: it receives a worldbody element
and a lane-local build context, appends its own bodies/geoms, and returns the
spawn, target and named features needed by an evaluator.  It must not read the
policy or the simulation loop.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..mathutils import local_to_world


@dataclass
class Feature:
    name: str
    kind: str
    geoms: tuple[str, ...] = ()
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    top_z: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def half_size(self) -> tuple[float, float, float]:
        return (self.size[0] / 2.0, self.size[1] / 2.0, self.size[2] / 2.0)


@dataclass
class ObstacleBuild:
    type: str
    spawn: tuple[float, float, float, float]
    terrain_origin: tuple[float, float, float]
    success_distance: float
    features: list[Feature] = field(default_factory=list)
    bounds: tuple[float, float, float, float] = (-5.0, 5.0, -2.0, 2.0)
    lane_width: float = 2.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional MJCF assets (a course module may bring meshes/textures).
    assets: list[ET.Element] = field(default_factory=list)

    def feature(self, name: str) -> Feature:
        for item in self.features:
            if item.name == name:
                return item
        raise KeyError(f"Obstacle '{self.type}' has no feature '{name}'")

    def features_of(self, kind: str) -> list[Feature]:
        return [item for item in self.features if item.kind == kind]

    def summary(self, feature_limit: int = 80) -> dict[str, Any]:
        """Compact JSON view; large course maps are truncated explicitly."""

        features = self.features[:feature_limit]
        return {
            "type": self.type,
            "spawn": list(self.spawn),
            "terrain_origin": list(self.terrain_origin),
            "success_distance": self.success_distance,
            "lane_width": self.lane_width,
            "feature_count": len(self.features),
            "features_truncated": len(self.features) > len(features),
            "features": [
                {
                    "name": f.name,
                    "kind": f.kind,
                    "center": list(f.center),
                    "size": list(f.size),
                    "top_z": f.top_z,
                }
                for f in features
            ],
            "metadata": dict(self.metadata),
        }


@dataclass
class BuildContext:
    worldbody: ET.Element
    tag: str
    origin: tuple[float, float, float]
    yaw: float
    difficulty: float
    rng: np.random.Generator
    robot_name: str = ""
    stand_height: float = 0.45
    wheel_radius: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    base_dir: Any = None

    def local(self, x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
        return local_to_world(self.origin, self.yaw, x, y, z)

    def body(self, name: str, pos: tuple[float, float, float] | None = None) -> ET.Element:
        body = ET.SubElement(
            self.worldbody,
            "body",
            {
                "name": f"{self.tag}_{name}",
                "pos": _fmt3(pos or self.origin),
                "euler": f"0 0 {self.yaw:.12g}",
            },
        )
        return body


class ObstacleModule(ABC):
    """A replaceable obstacle brick."""

    type: str = "base"
    label: str = "Base obstacle"
    default_params: dict[str, Any] = {}

    def params(self, difficulty: float, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(self.default_params)
        merged.update(overrides or {})
        merged["difficulty"] = float(np.clip(difficulty, 0.0, 1.0))
        return merged

    @abstractmethod
    def build(self, ctx: BuildContext) -> ObstacleBuild:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# XML helpers shared by modules
# ---------------------------------------------------------------------------

_TERRAIN_FRICTION = "1.0 0.005 0.0001"
_TERRAIN_COLOR = (0.46, 0.50, 0.54, 1.0)


def _fmt3(values: tuple[float, float, float]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def add_box(
    body: ET.Element,
    name: str,
    center: tuple[float, float, float],
    half_size: tuple[float, float, float],
    rgba: tuple[float, float, float, float] = _TERRAIN_COLOR,
    friction: str = _TERRAIN_FRICTION,
) -> str:
    ET.SubElement(
        body,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": _fmt3(center),
            "size": _fmt3(half_size),
            "rgba": " ".join(f"{v:.6g}" for v in rgba),
            "group": "0",
            "contype": "1",
            "conaffinity": "1",
            "friction": friction,
        },
    )
    return name


def add_cylinder(
    body: ET.Element,
    name: str,
    center: tuple[float, float, float],
    radius: float,
    half_length: float,
    rgba: tuple[float, float, float, float],
    quat: str | None = None,
) -> str:
    attrs = {
        "name": name,
        "type": "cylinder",
        "pos": _fmt3(center),
        "size": f"{radius:.9g} {half_length:.9g}",
        "rgba": " ".join(f"{v:.6g}" for v in rgba),
        "group": "0",
        "contype": "1",
        "conaffinity": "1",
        "friction": _TERRAIN_FRICTION,
    }
    if quat:
        attrs["quat"] = quat
    ET.SubElement(body, "geom", attrs)
    return name


def add_floor(
    body: ET.Element,
    name: str,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    top_z: float = 0.0,
    thickness: float = 0.20,
    rgba: tuple[float, float, float, float] = _TERRAIN_COLOR,
) -> str:
    return add_box(
        body,
        name,
        (center_xy[0], center_xy[1], top_z - thickness / 2.0),
        (size_xy[0] / 2.0, size_xy[1] / 2.0, thickness / 2.0),
        rgba,
    )


def add_box_between(
    body: ET.Element,
    name: str,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    z0: float,
    z1: float,
    rgba: tuple[float, float, float, float] = _TERRAIN_COLOR,
) -> str:
    if z1 <= z0:
        raise ValueError(f"{name}: z1 must be greater than z0")
    return add_box(
        body,
        name,
        (center_xy[0], center_xy[1], (z0 + z1) / 2.0),
        (size_xy[0] / 2.0, size_xy[1] / 2.0, (z1 - z0) / 2.0),
        rgba,
    )


def lerp(low: float, high: float, t: float) -> float:
    return float(low + (high - low) * float(np.clip(t, 0.0, 1.0)))


def polar(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
