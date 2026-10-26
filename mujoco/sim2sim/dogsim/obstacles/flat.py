"""Flat-ground brick: the baseline locomotion / standing test."""

from __future__ import annotations

from .base import BuildContext, Feature, ObstacleBuild, ObstacleModule, add_floor
from .registry import OBSTACLES


class FlatObstacle(ObstacleModule):
    type = "flat"
    label = "Flat ground"
    default_params = {"length": 10.0, "width": 4.0, "spawn_inset": 2.0, "success_distance": 4.0}

    def build(self, ctx: BuildContext) -> ObstacleBuild:
        p = self.params(ctx.difficulty, ctx.params)
        length = float(p["length"])
        width = float(p["width"])
        body = ctx.body("ground")
        add_floor(body, f"{ctx.tag}_floor", (0.0, 0.0), (length, width), top_z=0.0)

        spawn_local_x = -length / 2.0 + float(p["spawn_inset"])
        wx, wy, wz = ctx.local(spawn_local_x, 0.0, ctx.stand_height + 0.01)
        feature = Feature(
            name="floor",
            kind="floor",
            geoms=(f"{ctx.tag}_floor",),
            center=ctx.local(0.0, 0.0, 0.0),
            size=(length, width, 0.2),
            top_z=ctx.origin[2],
        )
        success = min(
            length - 2.0 * float(p["spawn_inset"]),
            float(p["success_distance"]),
        )
        return ObstacleBuild(
            type=self.type,
            spawn=(wx, wy, wz, ctx.yaw),
            terrain_origin=(ctx.origin[0], ctx.origin[1], ctx.origin[2]),
            success_distance=success,
            features=[feature],
            bounds=(-length / 2.0, length / 2.0, -width / 2.0, width / 2.0),
            lane_width=width,
            metadata={"length": length, "width": width, "spawn_inset": p["spawn_inset"]},
        )


FLAT = FlatObstacle()

OBSTACLES.register("flat", FLAT)
