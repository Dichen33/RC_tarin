"""Low-bar brick (optional ducking / clearance test)."""

from __future__ import annotations

import math

from .base import (
    BuildContext,
    Feature,
    ObstacleBuild,
    ObstacleModule,
    add_box_between,
    add_cylinder,
    lerp,
)
from .registry import OBSTACLES


class LowBarObstacle(ObstacleModule):
    type = "low_bar"
    label = "Low bars"
    default_params = {
        "length": 7.0,
        "width": 2.0,
        "num_bars": 6,
        "clearance_range": (0.34, 0.24),
        "bar_radius": 0.025,
        "bar_length": 1.60,
        "first_bar_x": -1.0,
        "spacing": 0.90,
    }

    def build(self, ctx: BuildContext) -> ObstacleBuild:
        p = self.params(ctx.difficulty, ctx.params)
        length = float(p["length"])
        width = float(p["width"])
        num_bars = int(p["num_bars"])
        clearance = lerp(p["clearance_range"][0], p["clearance_range"][1], ctx.difficulty)
        radius = float(p["bar_radius"])
        bar_length = float(p["bar_length"])
        first_x = float(p["first_bar_x"])
        spacing = float(p["spacing"])

        body = ctx.body("low_bar")
        add_box_between(
            body,
            f"{ctx.tag}_floor",
            (0.0, 0.0),
            (length + 2.0, width + 1.2),
            -0.20,
            0.0,
            rgba=(0.48, 0.48, 0.50, 1.0),
        )
        bar_z = clearance + radius
        features: list[Feature] = []
        for index in range(num_bars):
            x = first_x + index * spacing
            name = f"{ctx.tag}_bar_{index}"
            add_cylinder(
                body,
                name,
                (x, 0.0, bar_z),
                radius,
                bar_length / 2.0,
                (0.16, 0.32, 0.74, 1.0),
                quat=f"{math.cos(math.pi / 4):.12g} {math.sin(math.pi / 4):.12g} 0 0",
            )
            for post_index, y in enumerate((-bar_length / 2.0, bar_length / 2.0)):
                add_cylinder(
                    body,
                    f"{ctx.tag}_bar_{index}_post_{post_index}",
                    (x, y, bar_z / 2.0),
                    radius,
                    bar_z / 2.0,
                    (0.16, 0.32, 0.74, 1.0),
                )
            features.append(
                Feature(
                    name=f"bar_{index}",
                    kind="bar",
                    geoms=(name,),
                    center=ctx.local(x, 0.0, bar_z),
                    size=(2.0 * radius, bar_length, 2.0 * radius),
                    top_z=ctx.origin[2] + bar_z + radius,
                    meta={"x": x, "clearance": clearance, "radius": radius},
                )
            )

        spawn_x = first_x - 1.0
        wx, wy, wz = ctx.local(spawn_x, 0.0, ctx.stand_height + 0.01)
        return ObstacleBuild(
            type=self.type,
            spawn=(wx, wy, wz, ctx.yaw),
            terrain_origin=(ctx.origin[0], ctx.origin[1], ctx.origin[2]),
            success_distance=first_x + (num_bars - 1) * spacing + 1.0 - spawn_x,
            features=features,
            bounds=(-1.5, length / 2.0, -width / 2.0, width / 2.0),
            lane_width=width,
            metadata={
                "clearance": clearance,
                "bar_radius": radius,
                "bar_length": bar_length,
                "num_bars": num_bars,
                "first_bar_x": first_x,
                "spacing": spacing,
                "difficulty": ctx.difficulty,
            },
        )


LOW_BAR = LowBarObstacle()

OBSTACLES.register("low_bar", LOW_BAR)
