"""Compatibility re-export of the single terrain implementation.

This module used to be a full copy of ``minidog_native/terrain.py`` and had
already drifted once (its ``_add_course`` interpolated the cube size in the
opposite direction from the height).  One implementation, imported here, so the
two robot tasks cannot diverge again.
"""

from minidog_native.terrain import *  # noqa: F401,F403
from minidog_native.terrain import (  # noqa: F401
    FlatGapTerrainCfg,
    FlatHighWallTerrainCfg,
    LowBarTerrainCfg,
    PlumBlossomTerrainCfg,
    RoughPlumBlossomTerrainCfg,
    course_geometry_at,
)
