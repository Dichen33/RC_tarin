"""Shared obstacle registry.

Keeping the registry in its own module avoids a circular import between the
package ``__init__`` (which imports every brick) and the bricks themselves.
"""

from __future__ import annotations

from ..registry import Registry
from .base import ObstacleModule

OBSTACLES: Registry[ObstacleModule] = Registry("obstacle")
