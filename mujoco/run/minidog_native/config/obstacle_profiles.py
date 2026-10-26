"""Composable obstacle building blocks for the Go2W terrain course.

The course is a set of independent families that can be switched on and off per
run, so the same code base trains a single obstacle or any combination of them
without editing task files::

    MINIDOG_OBSTACLES=plum            # flat/rough warm-up, then stepping stones
    MINIDOG_OBSTACLES=stairs,gap      # two families, released in that order
    MINIDOG_OBSTACLES=all             # full course (default)
    MINIDOG_OBSTACLES=none            # locomotion only, no obstacle family

Flat, rough and perlin terrain are **always** kept: the course needs a warm-up
row, and the learning-progress sampler needs a history family to revisit.  The
environment only chooses which *obstacle* families exist; the release order and
the per-family success gates are defined here once and consumed by both the
terrain generator and the curriculum.
"""

from __future__ import annotations

import os
import re

# Terrain families.  A family is a group of sub-terrains that share one success
# gate, so a single switch can add or remove a coherent obstacle type.
BASE_TERRAIN_NAMES: tuple[str, ...] = ("flat", "random_rough", "perlin_noise")

OBSTACLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "stairs": ("pyramid_stairs", "pyramid_stairs_inv"),
    "blocks": ("random_grid",),
    "slope": ("sloped_terrain",),
    "gap": ("flat_gap",),
    "plum": ("rough_plum_blossom",),
    "wall": ("flat_high_wall",),
}

# Legacy groupings kept for callers that still reference them by name.
STANDARD_OBSTACLE_NAMES: tuple[str, ...] = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "random_grid",
    "sloped_terrain",
)
SPECIAL_OBSTACLE_NAMES: tuple[str, ...] = (
    "flat_gap",
    "rough_plum_blossom",
    "flat_high_wall",
)

# Release gates, in the order the curriculum unlocks them:
#   (gate name, families it covers, minimum episodes, success threshold, unlock row)
# A gate whose families are all disabled is skipped entirely, so a plum-only
# run releases plum blossom immediately after the base locomotion gate.
GATE_GROUPS: tuple[tuple[str, tuple[str, ...], int, float, int], ...] = (
    ("standard", ("stairs", "blocks", "slope"), 1024, 0.65, 3),
    ("gap", ("gap",), 512, 0.60, 4),
    ("plum", ("plum",), 512, 0.60, 6),
    ("wall", ("wall",), 512, 0.60, 8),
)

ENV_VAR = "MINIDOG_OBSTACLES"

# Convenient spellings: terrain names, plurals and synonyms all resolve to the
# family that owns them, so a run can be described with the name that is
# printed in the terrain config.
FAMILY_ALIASES: dict[str, str] = {
    "stair": "stairs",
    "pyramid_stairs": "stairs",
    "pyramid_stairs_inv": "stairs",
    "stairs_up": "stairs",
    "stairs_inv": "stairs",
    "inverted_stairs": "stairs",
    "block": "blocks",
    "grid": "blocks",
    "random_grid": "blocks",
    "sloped_terrain": "slope",
    "slopes": "slope",
    "ditch": "gap",
    "trench": "gap",
    "flat_gap": "gap",
    "plum_blossom": "plum",
    "stepping_stones": "plum",
    "stones": "plum",
    "rough_plum_blossom": "plum",
    "high_wall": "wall",
    "flat_high_wall": "wall",
    "walls": "wall",
}


def _resolve_token(token: str) -> str | None:
    """Map one token to a family name, or None when it is not recognised."""
    key = token.strip().lower().replace("-", "_")
    if key in OBSTACLE_FAMILIES:
        return key
    return FAMILY_ALIASES.get(key)


def selected_families(raw: str | None = None) -> tuple[str, ...]:
    """Return the enabled obstacle families in their canonical order.

    Accepts ``all``, ``none``, family names, terrain names and hyphenated
    combinations, separated by ``,`` ``;`` ``+`` ``/`` whitespace or ``-``::

        "plum"                 -> ("plum",)
        "stairs-blocks"        -> ("stairs", "blocks")
        "flat_high_wall,gap"   -> ("gap", "wall")
    """
    text = os.environ.get(ENV_VAR, "all") if raw is None else raw
    tokens = [
        token
        for token in re.split(r"[,;+/|\s]+", str(text).strip().lower())
        if token
    ]
    if not tokens or "all" in tokens:
        return tuple(OBSTACLE_FAMILIES)
    if "none" in tokens:
        return ()

    resolved: list[str] = []
    for token in tokens:
        family = _resolve_token(token)
        if family is None and "-" in token:
            # "stairs-blocks" -> two families; only accepted when every part is
            # a known name, otherwise the token is reported as unknown.
            parts = [_resolve_token(part) for part in token.split("-") if part]
            if parts and all(parts):
                resolved.extend(parts)  # type: ignore[arg-type]
                continue
        if family is None:
            raise ValueError(
                f"{ENV_VAR}: unknown obstacle family {token!r}. "
                f"Valid names: {sorted(OBSTACLE_FAMILIES)}, "
                f"aliases: {sorted(FAMILY_ALIASES)}, or 'all' / 'none'."
            )
        resolved.append(family)
    return tuple(name for name in OBSTACLE_FAMILIES if name in resolved)


def enabled_obstacle_names(raw: str | None = None) -> tuple[str, ...]:
    """Return every sub-terrain name contributed by the enabled families."""
    return tuple(
        name
        for family in selected_families(raw)
        for name in OBSTACLE_FAMILIES[family]
    )


def profile_name(raw: str | None = None) -> str:
    """Human readable profile label, e.g. ``plum+wall`` or ``none``."""
    families = selected_families(raw)
    return "+".join(families) if families else "none"


_announced: set[str] = set()


def announce_profile() -> str:
    """Print the active profile once per process and return its name."""
    name = profile_name()
    if name not in _announced:
        _announced.add(name)
        print(f"[Go2W] obstacle profile = {name}  ({ENV_VAR})", flush=True)
    return name
