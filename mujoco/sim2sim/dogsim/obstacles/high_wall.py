"""High-wall brick: three repeated transverse walls on flat ground."""

from __future__ import annotations

import numpy as np

from .base import BuildContext, Feature, ObstacleBuild, ObstacleModule, add_box_between, add_floor, lerp
from .registry import OBSTACLES


class HighWallObstacle(ObstacleModule):
    type = "high_wall"
    label = "High walls"
    default_params = {
        "length": 7.0,
        "width": 2.4,
        "wall_offsets": (1.40, 2.95, 4.50),
        "wall_height_range": (0.08, 0.42),
        "wall_start_difficulty": 0.10,
        "wall_thickness": 0.05,
        "wall_length": 1.80,
        "min_height": 0.06,
    }

    def build(self, ctx: BuildContext) -> ObstacleBuild:
        p = self.params(ctx.difficulty, ctx.params)
        d = ctx.difficulty
        length = float(p["length"])
        width = float(p["width"])
        offsets = tuple(float(v) for v in p["wall_offsets"])
        thickness = float(p["wall_thickness"])
        wall_length = float(p["wall_length"])
        start = float(p["wall_start_difficulty"])
        h_low, h_high = (float(v) for v in p["wall_height_range"])
        if d <= start:
            height = float(p["min_height"])
        else:
            progress = (d - start) / max(1.0 - start, 1.0e-6)
            height = lerp(h_low, h_high, progress)
        height = max(float(p["min_height"]), height)

        body = ctx.body("wall")
        add_floor(body, f"{ctx.tag}_floor", (1.0, 0.0), (length + 4.0, width + 1.2), top_z=0.0)
        features: list[Feature] = []
        for index, x in enumerate(offsets):
            name = f"{ctx.tag}_wall_{index}"
            add_box_between(
                body,
                name,
                (x, 0.0),
                (thickness, wall_length),
                -0.10,
                height,
                rgba=(0.72, 0.30, 0.16, 1.0),
            )
            features.append(
                Feature(
                    name=f"wall_{index}",
                    kind="wall",
                    geoms=(name,),
                    center=ctx.local(x, 0.0, height),
                    size=(thickness, wall_length, height + 0.10),
                    top_z=ctx.origin[2] + height,
                    meta={"x": x, "height": height, "half_length": wall_length / 2.0},
                )
            )

        spawn_x = -0.60
        wx, wy, wz = ctx.local(spawn_x, 0.0, ctx.stand_height + 0.01)
        return ObstacleBuild(
            type=self.type,
            spawn=(wx, wy, wz, ctx.yaw),
            terrain_origin=(ctx.origin[0], ctx.origin[1], ctx.origin[2]),
            success_distance=float(offsets[-1]) + 0.30,
            features=features,
            bounds=(-3.5, float(offsets[-1]) + 1.0, -width / 2.0, width / 2.0),
            lane_width=width,
            metadata={
                "height": height,
                "wall_offsets": list(offsets),
                "wall_thickness": thickness,
                "wall_length": wall_length,
                "difficulty": d,
            },
        )


HIGH_WALL = HighWallObstacle()

OBSTACLES.register("high_wall", HIGH_WALL)
OBSTACLES.register("wall", HIGH_WALL)
