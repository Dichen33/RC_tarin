"""Stair brick: a straight staircase with a flat top platform.

Step height is interpolated from the training range (0.04 -> 0.32 m) and the
tread is the training value (0.32 m).  This is the directional equivalent of
the pyramid-stairs terrain family used by the curriculum.
"""

from __future__ import annotations

from .base import BuildContext, Feature, ObstacleBuild, ObstacleModule, add_box_between, lerp
from .registry import OBSTACLES


class StairsObstacle(ObstacleModule):
    type = "stairs"
    label = "Stairs"
    default_params = {
        "num_steps": 4,
        "rise_range": (0.04, 0.32),
        "tread": 0.32,
        "platform_width": 1.20,
        "first_step_x": 0.80,
        "width": 2.2,
        "length": 8.0,
    }

    def build(self, ctx: BuildContext) -> ObstacleBuild:
        p = self.params(ctx.difficulty, ctx.params)
        d = ctx.difficulty
        num_steps = int(p["num_steps"])
        rise = lerp(p["rise_range"][0], p["rise_range"][1], d)
        tread = float(p["tread"])
        platform_width = float(p["platform_width"])
        first = float(p["first_step_x"])
        width = float(p["width"])
        length = float(p["length"])

        body = ctx.body("stairs")
        # The approach is a single slab from behind the spawn to the first riser.
        approach_start = -3.0
        add_box_between(
            body,
            f"{ctx.tag}_approach",
            ((approach_start + first) / 2.0, 0.0),
            (first - approach_start, width + 1.2),
            -0.20,
            0.0,
            rgba=(0.50, 0.50, 0.52, 1.0),
        )

        features: list[Feature] = []
        for index in range(num_steps):
            x0 = first + index * tread
            x1 = x0 + tread
            top = (index + 1) * rise
            name = f"{ctx.tag}_step_{index}"
            add_box_between(
                body,
                name,
                ((x0 + x1) / 2.0, 0.0),
                (tread, width),
                -0.20,
                top,
                rgba=(0.72, 0.45, 0.18, 1.0),
            )
            features.append(
                Feature(
                    name=f"step_{index}",
                    kind="step",
                    geoms=(name,),
                    center=ctx.local((x0 + x1) / 2.0, 0.0, top),
                    size=(tread, width, top + 0.20),
                    top_z=ctx.origin[2] + top,
                    meta={"index": index, "x0": x0, "x1": x1, "top": top},
                )
            )

        top_start = first + num_steps * tread
        top_end = top_start + platform_width
        top_z = num_steps * rise
        platform_name = f"{ctx.tag}_top_platform"
        add_box_between(
            body,
            platform_name,
            ((top_start + top_end) / 2.0, 0.0),
            (platform_width, width),
            -0.20,
            top_z,
            rgba=(0.24, 0.58, 0.72, 1.0),
        )
        features.append(
            Feature(
                name="top_platform",
                kind="platform",
                geoms=(platform_name,),
                center=ctx.local((top_start + top_end) / 2.0, 0.0, top_z),
                size=(platform_width, width, top_z + 0.20),
                top_z=ctx.origin[2] + top_z,
                meta={"x_start": top_start, "x_end": top_end, "top_z": top_z},
            )
        )

        spawn_x = -0.60
        wx, wy, wz = ctx.local(spawn_x, 0.0, ctx.stand_height + 0.01)
        return ObstacleBuild(
            type=self.type,
            spawn=(wx, wy, wz, ctx.yaw),
            terrain_origin=(ctx.origin[0], ctx.origin[1], ctx.origin[2]),
            success_distance=top_end + 0.30 - spawn_x,
            features=features,
            bounds=(-3.0, length / 2.0, -width / 2.0, width / 2.0),
            lane_width=width,
            metadata={
                "num_steps": num_steps,
                "rise": rise,
                "tread": tread,
                "top_z": top_z,
                "top_start": top_start,
                "top_end": top_end,
                "difficulty": d,
            },
        )


STAIRS = StairsObstacle()

OBSTACLES.register("stairs", STAIRS)
OBSTACLES.register("stair", STAIRS)
