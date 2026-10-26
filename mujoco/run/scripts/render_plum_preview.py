"""Render the MiniDog plum-blossom terrain without creating a training env."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from minidog_native.config.course_profiles import plum_course_terrain_generator
from minidog_native.terrain import FlatHighWallTerrainCfg


def render(model: mujoco.MjModel, data: mujoco.MjData, output: Path, *, top: bool) -> None:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = (4.0, 4.0, 0.05)
    if top:
        camera.distance = 6.5
        camera.azimuth = 90.0
        camera.elevation = -89.0
    else:
        camera.distance = 5.0
        camera.azimuth = 135.0
        camera.elevation = -32.0

    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    with mujoco.Renderer(model, height=900, width=1600) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        Image.fromarray(renderer.render()).save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated") / "plum_preview",
    )
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--terrain", choices=("plum", "wall"), default="plum")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 900
    spec.option.gravity = (0.0, 0.0, -9.81)
    spec.worldbody.add_body(name="terrain")
    if args.terrain == "wall":
        terrain = FlatHighWallTerrainCfg(size=(8.0, 8.0))
        output_name = "flat_high_wall"
    else:
        # Read the course from the profile builder.  The class defaults are a
        # different ladder from the one training compiles, so a preview built from
        # them used to show a course that never existed in an env.
        terrain = replace(
            plum_course_terrain_generator().sub_terrains["rough_plum_blossom"],
            size=(8.0, 8.0),
        )
        output_name = "plum_blossom"
    output = terrain.function(args.difficulty, spec, np.random.default_rng(2027))
    for index, geometry in enumerate(output.geometries):
        if geometry.geom is not None and geometry.color is not None:
            geometry.geom.rgba = geometry.color
            geometry.geom.name = f"plum_geom_{index}"

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    render(model, data, args.output_dir / f"{output_name}_overview.png", top=False)
    render(model, data, args.output_dir / f"{output_name}_top.png", top=True)
    print(f"Generated previews in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
