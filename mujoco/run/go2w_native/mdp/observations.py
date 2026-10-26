"""Privileged observations used by the Go2W value function.

The terrain context is intentionally critic-only.  It reduces value
interference between flat driving and obstacle traversal while keeping the
deployed actor dependent only on ODIN1-compatible observations.
"""

from __future__ import annotations

import torch


def terrain_context(env) -> torch.Tensor:
    """Return normalized terrain-family and difficulty identifiers."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None or terrain.terrain_origins is None:
        return torch.zeros((env.num_envs, 2), device=env.device)
    type_count = max(int(terrain.terrain_origins.shape[1]) - 1, 1)
    level_count = max(int(terrain.terrain_origins.shape[0]) - 1, 1)
    family = terrain.terrain_types.float() / float(type_count)
    level = terrain.terrain_levels.float() / float(level_count)
    return torch.stack((family.clamp(0.0, 1.0), level.clamp(0.0, 1.0)), dim=-1)
