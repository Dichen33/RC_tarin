"""Gap / ditch brick: a transverse ditch with a visible bottom."""

from __future__ import annotations

from .base import BuildContext, Feature, ObstacleBuild, ObstacleModule, add_box_between, add_floor, lerp
from .registry import OBSTACLES


class GapObstacle(ObstacleModule):
    type = "gap"
    label = "Gap / ditch"
    default_params = {
        "length": 7.0,
        "width": 2.4,
        "gap_start": 1.15,
        "gap_width_range": (0.10, 0.65),
        "gap_depth_range": (0.04, 0.45),
        "floor_thickness": 0.20,
        "bottom_thickness": 0.08,
        "success_distance": 3.35,
    }

    def build(self, ctx: BuildContext) -> ObstacleBuild:
        p = self.params(ctx.difficulty, ctx.params)
        d = ctx.difficulty
        length = float(p["length"])
        width = float(p["width"])
        gap_start = float(p["gap_start"])
        gap_width = lerp(p["gap_width_range"][0], p["gap_width_range"][1], d)
        gap_depth = lerp(p["gap_depth_range"][0], p["gap_depth_range"][1], d)
        gap_end = gap_start + gap_width
        floor_thickness = float(p["floor_thickness"])

        body = ctx.body("gap")
        # Approach and landing are separate boxes: there is deliberately no
        # hidden full-height floor under the ditch.
        approach_start = -3.0
        add_box_between(
            body,
            f"{ctx.tag}_approach",
            ((approach_start + gap_start) / 2.0, 0.0),
            (gap_start - approach_start, width + 1.2),
            -floor_thickness,
            0.0,
            rgba=(0.52, 0.52, 0.54, 1.0),
        )
        landing_end = length / 2.0 + 1.5
        add_box_between(
            body,
            f"{ctx.tag}_landing",
            ((gap_end + landing_end) / 2.0, 0.0),
            (landing_end - gap_end, width + 1.2),
            -floor_thickness,
            0.0,
            rgba=(0.52, 0.52, 0.54, 1.0),
        )
        add_box_between(
            body,
            f"{ctx.tag}_bottom",
            ((gap_start + gap_end) / 2.0, 0.0),
            (gap_width, width),
            -gap_depth - float(p["bottom_thickness"]),
            -gap_depth,
            rgba=(0.27, 0.27, 0.29, 1.0),
        )

        spawn_x = -0.60
        wx, wy, wz = ctx.local(spawn_x, 0.0, ctx.stand_height + 0.01)
        gap_feature = Feature(
            name="gap",
            kind="gap",
            geoms=(f"{ctx.tag}_bottom",),
            center=ctx.local((gap_start + gap_end) / 2.0, 0.0, -gap_depth / 2.0),
            size=(gap_width, width, gap_depth),
            top_z=ctx.origin[2],
            meta={"start_x": gap_start, "end_x": gap_end, "width": gap_width, "depth": gap_depth},
        )
        return ObstacleBuild(
            type=self.type,
            spawn=(wx, wy, wz, ctx.yaw),
            terrain_origin=(ctx.origin[0], ctx.origin[1], ctx.origin[2]),
            success_distance=float(p["success_distance"]),
            features=[gap_feature],
            bounds=(-3.0, length / 2.0 + 1.5, -width / 2.0, width / 2.0),
            lane_width=width,
            metadata={
                "gap_start": gap_start,
                "gap_end": gap_end,
                "gap_width": gap_width,
                "gap_depth": gap_depth,
                "difficulty": d,
            },
        )


GAP = GapObstacle()

OBSTACLES.register("gap", GAP)
OBSTACLES.register("ditch", GAP)
