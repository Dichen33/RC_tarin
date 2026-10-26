"""Reusable terrain profiles for the MiniDog obstacle curriculum.

The Go2 reference keeps terrain composition in one task configuration.  This
module applies the same rule to the native MJLab task, while leaving the
custom MuJoCo geometries in ``terrain.py`` unchanged.
"""

from __future__ import annotations

import os

from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    HfPerlinNoiseTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    TerrainGeneratorCfg,
)

from ..terrain import FlatGapTerrainCfg, FlatHighWallTerrainCfg, RoughPlumBlossomTerrainCfg


PLUM_SOFT_ENV_VAR = "GO2W_PLUM_SOFT"


def plum_soft() -> bool:
    """BeamDojo stage-1 switch: fill the same-row gaps on the *physical* terrain.

    The fill boxes are added to MuJoCo geom group 1, while the ray-cast height
    scanner reads group 0 only (``include_geom_groups=(0,)``).  So the physics
    becomes a continuous surface -- a misstep can no longer drop the robot -- while
    the perception still shows the real stones.  That is BeamDojo's stage-1
    decoupling (arXiv 2502.10363, Sec. IV-C1: "trained on flat terrain while being
    provided with a corresponding height map of the true task terrains ... missteps
    are penalized but do not lead to termination"):

    * the foothold reward (``plum_foothold`` / ``plum_overhang``) is scored against
      the real stone layout, so it can only be earned by *reading the radar*;
    * exploring a placement no longer ends the episode, so safe foothold samples --
      the thing that is nearly impossible to collect with hard termination -- become
      abundant.

    Default off: it changes the task, so it must be an explicit opt-in.
    """
    return os.environ.get(PLUM_SOFT_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


from .obstacle_profiles import (
    BASE_TERRAIN_NAMES,
    SPECIAL_OBSTACLE_NAMES,
    STANDARD_OBSTACLE_NAMES,
    announce_profile,
    enabled_obstacle_names,
)

# ROBOCON2027.xml is treated as a medium reference, not as the final limit.
# The terrain rows start below these values and the last rows deliberately go
# beyond them so that a successful policy can keep improving.
XML_MEDIUM_OBSTACLE = {
    "step_height": 0.24,
    "stepping_stone_size": 0.20,
    "gap_width": 0.42,
    "gap_depth": 0.22,
    "plum_block_height": 0.20,
    "high_wall_height": 0.30,
}


def plum_course_terrain_generator() -> TerrainGeneratorCfg:
    """Build the complete grid used by ``Robot-PlumBlossom-v0``.

    Curriculum mode creates one column per named terrain and one difficulty
    row per level.  The curriculum manager decides which columns may be
    sampled; all columns remain compiled so level changes do not rebuild the
    MuJoCo model during training.
    """

    size = (8.0, 8.0)
    sub_terrains = {
            "flat": BoxFlatTerrainCfg(proportion=0.10, size=size),
            "pyramid_stairs": BoxPyramidStairsTerrainCfg(
                # ROBOCON XML medium is about 0.24 m.  The hardest row stops at
                # 0.32 m (1.33x medium): with the body held at stand height the
                # Go2W can lift one wheel by at most 0.21 m, so a 0.40 m step is
                # not steppable and only wastes curriculum attempts.
                proportion=0.10, step_height_range=(0.04, 0.32), step_width=0.32, size=size
            ),
            "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.08, step_height_range=(0.04, 0.36), step_width=0.32, size=size
            ),
            "random_grid": BoxRandomGridTerrainCfg(
                # 0.30 m tiles were 26x26 = 676 boxes per patch and 60% of the
                # whole terrain budget.  0.60 m tiles keep a genuine block field
                # (fine roughness is already covered by ``random_rough``) at a
                # quarter of the collision geoms.
                proportion=0.10, grid_width=0.60, grid_height_range=(0.02, 0.30), size=size
            ),
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=0.08,
                noise_range=(0.0, 0.04),
                noise_step=0.01,
                horizontal_scale=0.16,
                downsampled_scale=0.16,
                border_width=0.25,
                base_thickness_ratio=100.0,
                size=size,
            ),
            "perlin_noise": HfPerlinNoiseTerrainCfg(
                proportion=0.07,
                height_range=(0.0, 0.04),
                octaves=2,
                persistence=0.4,
                lacunarity=2.0,
                horizontal_scale=0.16,
                resolution=0.16,
                border_width=0.25,
                base_thickness_ratio=100.0,
                size=size,
            ),
            "sloped_terrain": HfPyramidSlopedTerrainCfg(
                proportion=0.05,
                slope_range=(0.03, 0.18),
                platform_width=2.0,
                border_width=0.25,
                base_thickness_ratio=100.0,
                horizontal_scale=0.16,
                size=size,
            ),
            "flat_gap": FlatGapTerrainCfg(
                proportion=0.12,
                gap_start_x=2.35,
                # 0.42 m / 0.22 m is the ROBOCON medium checkpoint.  The hard
                # rows stop at 0.65 m: clearing a wider ditch needs a jump of
                # that length, and the platform has neither the body length nor
                # a trained jump skill yet.  Depth still reaches 0.45 m.
                gap_width_range=(0.10, 0.65),
                gap_depth_range=(0.04, 0.45),
            ),
            "rough_plum_blossom": RoughPlumBlossomTerrainCfg(
                proportion=0.22,
                # Ranges are (easy, hard): index 0 at difficulty 0, index 1 at
                # difficulty 1.  Cube tops stay flush with the platforms because
                # block_height_range[1] == platform_height under the
                # 0.5 + 0.5 * d platform ramp.
                #
                # The easy end is 0.60 m, deliberately a *solid slab*: at 0.60 m
                # the same-row pitch (0.60 m) and the row separation (0.40 m) are
                # both fully covered, so row 0 is one continuous surface between
                # the platforms.  That is the bootcamp row the robot can drive and
                # roll across before it ever has to bridge a gap.
                #
                # The 7-row ramp then interpolates
                #   0.600 / 0.533 / 0.467 / 0.400 / 0.333 / 0.267 / 0.200
                # i.e. same-row gaps 0.000 / 0.067 / 0.133 / 0.200 / 0.267 /
                # 0.333 / 0.400 m.  The first three are narrower than the wheel
                # diameter (0.172 m), so a wheel bridges them; row 3 is the first
                # gap a wheel can actually drop into.  The previous easy end
                # (0.376 m) already had a 0.224 m gap, larger than the wheel:
                # there was no easy row at all, and the measured policy never left
                # rows 0-1 because nothing on them was learnable.
                block_size_range=(0.60, 0.20),     # same-row gap 0.000 m -> 0.400 m
                block_height_range=(0.10, 0.20),   # 0.10 m -> 0.20 m
                rough_grid_width=0.85,
                platform_width=0.80,
                platform_height=0.20,              # 0.10 m -> 0.20 m
                # The platform-to-cube slots under each wheel line are bridged on
                # the easy rows and fade out linearly: 0.00 / 0.04 / 0.09 / 0.15 /
                # 0.22 / 0.31 / 0.40 m on rows 0-6.  Rows 0-2 therefore really are
                # rollable, as the ladder above claims, and row 3 keeps a 0.15 m
                # entry slot (still narrower than the 0.172 m wheel) while its
                # same-row gaps are the first ones a wheel can drop into.
                bridge_fade_end=1.0,
                # Shaped size ramp: 0.600 / 0.557 / 0.499 / 0.435 / 0.357 / 0.282 / 0.200 m
                # -> same-row wheel slots 0.000 / 0.043 / 0.101 / 0.170 / 0.243 /
                # 0.318 / 0.400 m.  Row 3 lands on the 0.172 m wheel diameter, so the
                # ladder has a rung for the "roll -> lift" transition instead of
                # jumping 0.133 -> 0.200 m in one step (plum_pilot3 solved row 2 with
                # success 1.00 and never crossed row 3 in 1000 iterations).  Row 0 and
                # row 6 keep the designed 0.600 m and competition 0.200 m endpoints.
                size_ramp_exponent=1.28,
                # The competition course sits on a flat <geom type="plane">:
                # there is no pit and no corridor.  Removing them also removes
                # the "belly hits the pit edge" termination that ended ~73% of
                # the plum episodes in the previous run.
                rough_height_range=(0.0, 0.0),
                course_pit_depth=0.0,
                # Every row now holds a course, so the release threshold is 0 and
                # no episode can land on a course-free row.
                course_start_difficulty=0.0,
                # BeamDojo stage 1 (`GO2W_PLUM_SOFT=1`): the same-row gaps are
                # physically filled in geom group 1, which the height scanner does
                # not read.  Physics becomes non-droppable while the radar still
                # shows the real stones, so the only way to earn the foothold
                # reward is to read the radar.  Default off.
                soft_fill=plum_soft(),
                # Per-patch layout jitter.  With a perfectly regular course the
                # policy can memorise one fixed plan and never needs its radar, so
                # the terrain observation carries no gradient.  +-1 cm position and
                # +-4 mm height keep the course inside the existing reward
                # tolerances (the foothold test allows +-6 cm around each cube) and
                # inside ``course_span`` (2 cm margin), while forcing the policy to
                # read the stones from the radar to place a wheel.
                #
                # 2026-10-25: leaving this at 1 cm.  Raising it to 2 cm was tried to
                # force look-ahead, and the audit rejected it: the same-row slot of
                # row 3 is 0.170 m nominal, i.e. already at the 0.172 m wheel
                # diameter, so +-2 cm of placement jitter opens it to as much as
                # 0.21 m and destroys the "rows 0-3 are rollable" invariant the
                # whole ladder depends on.  Look-ahead is bought through
                # ``plum_next_foothold`` (0.5 -> 1.5) instead, which pays for
                # approaching the next stone without perturbing the geometry.
                position_jitter=0.01,
                height_jitter=0.004,
            ),
            "flat_high_wall": FlatHighWallTerrainCfg(
                proportion=0.10,
                # ROBOCON wall is about 0.30 m high.  Clearing a wall of height
                # h needs the base at h + 0.10 m (legs tucked), so 0.42 m is the
                # practical ceiling: it already asks for ~0.10 m of jump above
                # the stand height, and 0.50 m asked for 0.19 m.
                wall_start_difficulty=0.10,
                wall_height_range=(0.08, 0.42),
                wall_length_frac=1.0,
                wall_thickness=0.05,
                wall_centers_x=(2.9, 4.45, 6.0),
            ),
    }

    # Building blocks: keep only the families this run selected.  Flat and rough
    # are always present (warm-up row plus the history family the
    # learning-progress sampler revisits), so a plum-only run still starts on
    # plain locomotion.
    keep = set(BASE_TERRAIN_NAMES) | set(enabled_obstacle_names())
    sub_terrains = {name: cfg for name, cfg in sub_terrains.items() if name in keep}
    if not sub_terrains:
        raise ValueError(
            "Go2W course became empty; check MINIDOG_OBSTACLES (flat/rough terrain "
            "must stay reachable)."
        )
    announce_profile()

    return TerrainGeneratorCfg(
        size=size,
        border_width=1.0,
        # One column per enabled terrain and 7 difficulty rows.  10 rows x 10
        # columns compiled 11,205 collision geoms into the single MuJoCo model,
        # so every physics substep swept the whole set for all 2048 worlds
        # (12 s / iteration, 4.1k steps/s).  Fewer rows and coarser grids cut
        # that by ~4x while keeping the medium-to-extra-hard span.
        #
        # 7 rows (not 6) because the plum-blossom release gate unlocks level 6,
        # and the bootcamp needs three rows below the first real gap: row 3 is
        # the first 0.200 m gap, so rows 0-2 are rollable warm-up rows.
        num_rows=7,
        num_cols=len(sub_terrains),
        curriculum=True,
        sub_terrains=sub_terrains,
    )
