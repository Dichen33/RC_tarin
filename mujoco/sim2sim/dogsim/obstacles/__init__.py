"""Obstacle bricks.

Importing this package imports every module once, which populates the shared
registry.  To add a brick, drop a module next to this file, register it in
``dogsim.obstacles.registry.OBSTACLES`` and import it below.
"""

from __future__ import annotations

from .base import BuildContext, Feature, ObstacleBuild, ObstacleModule
from .registry import OBSTACLES

# Import for the registration side effect (keep the order readable).
from .flat import FLAT  # noqa: F401,E402
from .rough import ROUGH  # noqa: F401,E402
from .plum import PLUM  # noqa: F401,E402
from .high_wall import HIGH_WALL  # noqa: F401,E402
from .gap import GAP  # noqa: F401,E402
from .stairs import STAIRS  # noqa: F401,E402
from .low_bar import LOW_BAR  # noqa: F401,E402
from .course import COURSE  # noqa: F401,E402


def get_obstacle(name: str) -> ObstacleModule:
    return OBSTACLES.get(name)


def obstacle_names() -> tuple[str, ...]:
    return OBSTACLES.names()


__all__ = [
    "BuildContext",
    "Feature",
    "ObstacleBuild",
    "ObstacleModule",
    "OBSTACLES",
    "get_obstacle",
    "obstacle_names",
    "FLAT",
    "ROUGH",
    "PLUM",
    "HIGH_WALL",
    "GAP",
    "STAIRS",
    "LOW_BAR",
    "COURSE",
]
