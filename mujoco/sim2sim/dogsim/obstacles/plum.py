"""Plum-blossom stepping-stone brick.

The layout matches ``RoughPlumBlossomTerrainCfg`` in the training project:
two staggered rows of three blocks, an entrance platform at the origin and an
exit platform at +2.7 m.  Block size and height are interpolated with the same
difficulty mapping as training.
"""

from __future__ import annotations

from .base import (
    BuildContext,
    Feature,
    ObstacleBuild,
    ObstacleModule,
    add_box_between,
    add_floor,
    lerp,
)
from .registry import OBSTACLES

# Relative to the entrance platform origin, matching the training reward.
BLOCK_CENTERS = (
    (0.6, 0.2), (1.2, 0.2), (1.8, 0.2),
    (0.9, -0.2), (1.5, -0.2), (2.1, -0.2),
)


class PlumObstacle(ObstacleModule):
    type = "plum"
    label = "Plum-blossom stepping stones"
    default_params = {
        "block_size_range": (0.36, 0.12),
        "block_height_range": (0.10, 0.32),
        "platform_height": 0.28,
        "platform_width": 0.80,
        "exit_x": 2.70,
        "catch_depth": 0.60,
        # Training's RoughPlumBlossomTerrainCfg puts the course on top of a
        # rough grid with a 1.10 m corridor and pits outside it.  Reproducing
        # that underlay matters: the actor's height encoder learned to expect
        # rough ground around the stones, not a deep empty box.
        "rough_underlay": True,
        "rough_grid_width": 0.85,
        "rough_height_range": (0.0, 0.10),
        "course_corridor_width": 1.10,
        "course_pit_depth": 0.35,
        "course_start_difficulty": 0.15,
        "rough_amplitude": 0.0,
    }

    def build(self, ctx: BuildContext) -> ObstacleBuild:
        p = self.params(ctx.difficulty, ctx.params)
        d = ctx.difficulty
        block_size = lerp(p["block_size_range"][1], p["block_size_range"][0], d)
        block_height = lerp(p["block_height_range"][0], p["block_height_range"][1], d)
        platform_height = float(p["platform_height"]) * (0.5 + 0.5 * d)
        platform_width = float(p["platform_width"])
        exit_x = float(p["exit_x"])
        catch_depth = float(p["catch_depth"])

        body = ctx.body("plum")
        features: list[Feature] = []
        add_floor(
            body,
            f"{ctx.tag}_catch",
            ((exit_x) / 2.0, 0.0),
            (exit_x + platform_width + 4.0, 5.0),
            top_z=-catch_depth,
            thickness=0.10,
            rgba=(0.24, 0.24, 0.26, 1.0),
        )

        rough_heights: list[float] = []
        if bool(p.get("rough_underlay", True)):
            grid = float(p["rough_grid_width"])
            h_low, h_high = (float(v) for v in p.get("rough_height_range", (0.0, 0.10)))
            # Training samples a scalar rough height per difficulty row and a
            # uniform +/- perturbation per grid cell.
            row_height = h_low + d * (h_high - h_low)
            if float(p.get("rough_amplitude", 0.0)) > 0.0:
                row_height = float(p["rough_amplitude"])
            x0, x1 = -2.6, exit_x + 2.2
            y0, y1 = -2.6, 2.6
            nx = max(1, int(round((x1 - x0) / grid)))
            ny = max(1, int(round((y1 - y0) / grid)))
            cell_x = (x1 - x0) / nx
            cell_y = (y1 - y0) / ny
            corridor = float(p["course_corridor_width"])
            pit = -abs(float(p["course_pit_depth"]))
            course_active = d >= float(p["course_start_difficulty"])
            for ix in range(nx):
                for iy in range(ny):
                    x = x0 + (ix + 0.5) * cell_x
                    y = y0 + (iy + 0.5) * cell_y
                    h = float(ctx.rng.uniform(-row_height, row_height))
                    if course_active and abs(y) > corridor / 2.0:
                        h = pit
                    rough_heights.append(h)
                    add_box_between(
                        body,
                        f"{ctx.tag}_rough_{ix:02d}_{iy:02d}",
                        (x, y),
                        (cell_x * 0.995, cell_y * 0.995),
                        -0.60,
                        h,
                        rgba=(0.42, 0.44, 0.46, 1.0),
                    )
        if rough_heights:
            features.append(
                Feature(
                    name="rough_underlay",
                    kind="rough",
                    center=ctx.local((exit_x) / 2.0, 0.0, 0.0),
                    size=(exit_x + 4.8, 5.2, max(rough_heights) - min(rough_heights)),
                    top_z=ctx.origin[2] + max(rough_heights),
                    meta={
                        "max_height": max(rough_heights),
                        "min_height": min(rough_heights),
                        "grid_width": float(p["rough_grid_width"]),
                        "corridor_width": float(p["course_corridor_width"]),
                        "pit_depth": float(p["course_pit_depth"]),
                    },
                )
            )
        for name, x in (("entry", 0.0), ("exit", exit_x)):
            add_box_between(
                body,
                f"{ctx.tag}_platform_{name}",
                (x, 0.0),
                (platform_width, platform_width),
                -0.20,
                platform_height,
                rgba=(0.24, 0.58, 0.26, 1.0),
            )

        for index, (bx, by) in enumerate(BLOCK_CENTERS):
            name = f"{ctx.tag}_block_{index}"
            add_box_between(
                body,
                name,
                (bx, by),
                (block_size, block_size),
                -0.20,
                block_height,
                rgba=(0.36, 0.66, 0.28, 1.0),
            )
            features.append(
                Feature(
                    name=f"block_{index}",
                    kind="stone",
                    geoms=(name,),
                    center=ctx.local(bx, by, block_height),
                    size=(block_size, block_size, block_height + 0.20),
                    top_z=ctx.origin[2] + block_height,
                    meta={
                        "row": 0 if by > 0.0 else 1,
                        "local": [bx, by],
                        "contact_radius": block_size / 2.0 + 0.06,
                    },
                )
            )

        features.append(
            Feature(
                name="entry_platform",
                kind="platform",
                geoms=(f"{ctx.tag}_platform_entry",),
                center=ctx.local(0.0, 0.0, platform_height),
                size=(platform_width, platform_width, platform_height + 0.20),
                top_z=ctx.origin[2] + platform_height,
            )
        )
        features.append(
            Feature(
                name="exit_platform",
                kind="platform",
                geoms=(f"{ctx.tag}_platform_exit",),
                center=ctx.local(exit_x, 0.0, platform_height),
                size=(platform_width, platform_width, platform_height + 0.20),
                top_z=ctx.origin[2] + platform_height,
            )
        )

        spawn_z = ctx.origin[2] + platform_height + ctx.stand_height + 0.01
        wx, wy, _ = ctx.local(0.0, 0.0, 0.0)
        return ObstacleBuild(
            type=self.type,
            spawn=(wx, wy, spawn_z, ctx.yaw),
            terrain_origin=(ctx.origin[0], ctx.origin[1], ctx.origin[2] + platform_height),
            success_distance=exit_x,
            features=features,
            bounds=(-1.0, exit_x + 1.0, -1.2, 1.2),
            lane_width=2.4,
            metadata={
                "block_size": block_size,
                "block_height": block_height,
                "platform_height": platform_height,
                "block_centers": [list(c) for c in BLOCK_CENTERS],
                "difficulty": d,
            },
        )


PLUM = PlumObstacle()

OBSTACLES.register("plum", PLUM)
OBSTACLES.register("plum_blossom", PLUM)
