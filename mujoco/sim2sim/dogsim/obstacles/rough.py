"""Rough-ground brick: a grid of sharp-edged blocks with increasing height."""

from __future__ import annotations

import numpy as np

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


class RoughObstacle(ObstacleModule):
    type = "rough"
    label = "Rough ground (random block grid)"
    default_params = {
        "length": 7.0,
        "width": 2.8,
        "cell": 0.40,
        # Easy row: 0-8 cm blocks.  Hard row matches the training
        # BoxRandomGridTerrainCfg range (2-30 cm).
        "height_range": (0.0, 0.08),
        "height_range_hard": (0.02, 0.30),
        "apron": 1.0,
        "rough_fraction": 0.75,
    }

    def build(self, ctx: BuildContext) -> ObstacleBuild:
        p = self.params(ctx.difficulty, ctx.params)
        length = float(p["length"])
        width = float(p["width"])
        cell = float(p["cell"])
        apron = float(p["apron"])
        easy_low, easy_high = (float(v) for v in p["height_range"])
        hard_low, hard_high = (float(v) for v in p["height_range_hard"])
        h_low = lerp(easy_low, hard_low, ctx.difficulty)
        h_high = lerp(easy_high, hard_high, ctx.difficulty)
        if h_low > h_high:
            raise ValueError("rough.height_range must be ascending")
        body = ctx.body("rough")
        flat_case = abs(h_low) < 1.0e-9 and abs(h_high) < 1.0e-9
        # The base slab is deliberately wider/longer than the lane so the
        # height scan never sees a void at the spawn; the evaluator still
        # rejects leaving the lane.  At zero roughness the slab is the only
        # surface: a coincident second box would create duplicate contacts.
        if not flat_case:
            add_floor(
                body,
                f"{ctx.tag}_base",
                (0.0, 0.0),
                (max(length, 12.0), width + 1.2),
                top_z=0.0,
                thickness=0.20,
            )

        x0 = -length / 2.0 + apron
        x1 = length / 2.0 - apron * 0.5
        y0 = -width / 2.0 + 0.10
        y1 = width / 2.0 - 0.10
        nx = max(1, int(round((x1 - x0) / cell)))
        ny = max(1, int(round((y1 - y0) / cell)))
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny
        heights: list[float] = []
        geoms: list[str] = []
        if flat_case:
            # Zero roughness is flat ground.  Building zero-height cells here
            # would add artificial seams and a lip, which is not what the
            # training generator does at difficulty 0.
            name = f"{ctx.tag}_flat_patch"
            add_floor(
                body,
                name,
                (0.0, 0.0),
                (max(length, 12.0), width + 1.2),
                top_z=0.0,
                thickness=0.20,
            )
            geoms.append(name)
        else:
            for ix in range(nx):
                for iy in range(ny):
                    h = float(ctx.rng.uniform(h_low, h_high))
                    heights.append(h)
                    cx = x0 + (ix + 0.5) * dx
                    cy = y0 + (iy + 0.5) * dy
                    name = f"{ctx.tag}_cell_{ix:02d}_{iy:02d}"
                    # Cells almost touch: a 0.4 m grid with 8% gaps creates
                    # crevices that are not part of the training heightfield.
                    add_box_between(
                        body,
                        name,
                        (cx, cy),
                        (dx * 0.995, dy * 0.995),
                        -0.10,
                        h,
                        rgba=(0.44 + 0.20 * h, 0.46 + 0.16 * h, 0.48 + 0.12 * h, 1.0),
                    )
                    geoms.append(name)

        spawn_x = -length / 2.0 + apron * 0.35
        wx, wy, wz = ctx.local(spawn_x, 0.0, ctx.stand_height + 0.01)
        feature = Feature(
            name="rough_patch",
            kind="rough",
            geoms=tuple(geoms),
            center=ctx.local((x0 + x1) / 2.0, 0.0, 0.0),
            size=(x1 - x0, y1 - y0, max(heights) if heights else 0.0),
            top_z=ctx.origin[2] + (max(heights) if heights else 0.0),
            meta={
                "cells": len(geoms),
                "grid": [nx, ny],
                "max_height": max(heights) if heights else 0.0,
                "mean_height": float(np.mean(heights)) if heights else 0.0,
                "height_range": [h_low, h_high],
            },
        )
        # Success is measured at the end of the rough patch, not at a fixed
        # global coordinate, so moving the lane never changes the metric.
        # The training default success distance for non-specific terrain is
        # 4.0 m.  Asking for the full patch length would make a 15 s smoke run
        # time out even at the flat 0.4 m/s wheel speed.
        success = min((x1 - spawn_x) - 0.6, 4.0)
        return ObstacleBuild(
            type=self.type,
            spawn=(wx, wy, wz, ctx.yaw),
            terrain_origin=(ctx.origin[0], ctx.origin[1], ctx.origin[2]),
            success_distance=success,
            features=[feature],
            bounds=(-length / 2.0, length / 2.0, -width / 2.0, width / 2.0),
            lane_width=width,
            metadata={
                "length": length,
                "width": width,
                "cell": cell,
                "grid": [nx, ny],
                "height_range": [h_low, h_high],
                "max_height": feature.meta["max_height"],
            },
        )


ROUGH = RoughObstacle()

OBSTACLES.register("rough", ROUGH)
