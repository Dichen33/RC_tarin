"""Integrated ROBOCON course brick.

This is a compatibility module for the original supplied course.  The five
evaluated obstacle families use their own focused bricks; ``course`` is for an
end-to-end route or a visual sanity check.  It copies the course assets and
worldbody into the composed scene and exposes named features for diagnostics.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT, resolve_path
from .base import BuildContext, Feature, ObstacleBuild, ObstacleModule
from .registry import OBSTACLES


def _radianize(node: ET.Element) -> None:
    compiler = node.find("compiler")
    angle = (compiler.get("angle", "degree") if compiler is not None else "degree").lower()
    if angle not in {"degree", "degrees"}:
        # ``robocon2027.v2.xml`` already declares angle="radian"; converting
        # its 1.5708 / 3.14159 euler values again would shrink every rotation
        # to ~1 degree and move the walls/stairs to the wrong orientation.
        return
    for child in node.iter():
        if "euler" in child.attrib:
            values = [float(v) for v in child.attrib["euler"].split()]
            child.set("euler", " ".join(f"{math.radians(v):.12g}" for v in values))
        if "axisangle" in child.attrib:
            values = [float(v) for v in child.attrib["axisangle"].split()]
            values[-1] = math.radians(values[-1])
            child.set("axisangle", " ".join(f"{v:.12g}" for v in values))


def _resolve_assets(course: ET.Element, course_xml: Path) -> list[ET.Element]:
    compiler = course.find("compiler")
    meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    texturedir = Path(compiler.get("texturedir", "")) if compiler is not None else Path()
    asset = course.find("asset")
    if asset is None:
        return []
    resolved: list[ET.Element] = []
    for node in list(asset):
        file_name = node.get("file")
        if file_name and not Path(file_name).is_absolute():
            base = course_xml.parent
            if node.tag == "mesh":
                base = base / meshdir
            elif node.tag == "texture":
                base = base / texturedir
            candidate = (base / file_name).resolve()
            if candidate.exists():
                node.set("file", candidate.as_posix())
            elif node.tag == "hfield" and not any(
                geom.get("hfield") == node.get("name") for geom in course.iter("geom")
            ):
                # The supplied map declares an unused hfield whose PNG is not
                # present.  It has no geoms, so dropping it is safe.
                continue
            else:
                raise FileNotFoundError(f"Course asset not found: {candidate}")
        resolved.append(copy.deepcopy(node))
    return resolved


def _feature_from_geom(geom: ET.Element) -> Feature | None:
    name = geom.get("name", "")
    if not name:
        return None
    kind = "course"
    if name.startswith("high_wall"):
        kind = "wall"
    elif name.startswith("stepping_stones"):
        kind = "stone"
    elif name.startswith("stairs"):
        kind = "step"
    elif name.startswith("platform"):
        kind = "platform"
    elif name.startswith("limit_frame"):
        kind = "frame"
    elif name.startswith("L_pile"):
        kind = "pile"
    elif name.startswith("tire_ring"):
        kind = "tire"
    elif name.startswith("pit_wall") or name.startswith("gravel"):
        kind = "pit"
    elif name.startswith("L_reach"):
        kind = "marker"
    pos = [float(v) for v in geom.get("pos", "0 0 0").split()]
    size = [float(v) for v in geom.get("size", "0 0 0").split()]
    if geom.get("type") == "cylinder":
        size = [size[0], size[0], size[1]]
    if geom.get("type") == "mesh":
        size = [0.0, 0.0, 0.0]
    # Rotate the 2D footprint for a 90-degree yaw so the map shows a
    # transverse wall as transverse instead of rotated by 90 degrees.
    if "euler" in geom.attrib and len(size) >= 2:
        try:
            euler = [float(v) for v in geom.attrib["euler"].split()]
            if len(euler) >= 3 and abs(math.sin(euler[2])) > 0.5:
                size[0], size[1] = size[1], size[0]
        except ValueError:
            pass
    return Feature(
        name=name,
        kind=kind,
        geoms=(name,),
        center=(pos[0], pos[1], pos[2]),
        size=(size[0] * 2.0, size[1] * 2.0, size[2] * 2.0),
        top_z=pos[2] + size[2],
        meta={"type": geom.get("type", "box"), "euler": geom.get("euler", "")},
    )


class CourseObstacle(ObstacleModule):
    type = "course"
    label = "ROBOCON integrated course"
    default_params = {
        "xml": "../../robocon2027.v2.xml",
        "spawn": [0.0, 0.0, 0.45, 0.0],
        "success_distance": 4.0,
    }

    def build(self, ctx: BuildContext) -> ObstacleBuild:
        p = self.params(ctx.difficulty, ctx.params)
        configured = p.get("xml")
        candidates = []
        if configured:
            candidates.append(Path(str(configured)).expanduser())
        # Search a few ancestors so a copied project still finds the supplied
        # map without hard-coding a machine-specific absolute path.
        for parent in (PROJECT_ROOT, PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent, PROJECT_ROOT.parent.parent.parent):
            candidates.append(parent / "ROBOCON2027.xml")
        course_xml: Path | None = None
        for candidate in candidates:
            resolved = resolve_path(candidate, ctx.base_dir) if ctx.base_dir is not None else candidate
            if resolved is not None and resolved.exists():
                course_xml = resolved
                break
        if course_xml is None:
            raise FileNotFoundError(f"Course XML not found; tried {candidates}")
        course = ET.parse(course_xml).getroot()
        _radianize(course)
        assets = _resolve_assets(course, course_xml)
        worldbody = course.find("worldbody")
        if worldbody is None:
            raise ValueError(f"Course has no worldbody: {course_xml}")
        features: list[Feature] = []
        for child in list(worldbody):
            cloned = copy.deepcopy(child)
            ctx.worldbody.append(cloned)
            for geom in cloned.iter("geom"):
                geom.set("group", "0")
                feature = _feature_from_geom(geom)
                if feature is not None:
                    features.append(feature)
        spawn = tuple(float(v) for v in p.get("spawn", (0.0, 0.0, ctx.stand_height, 0.0)))
        if len(spawn) != 4:
            raise ValueError("course.spawn must contain x y z yaw")
        bounds = (-8.0, 8.0, -6.0, 6.0)
        return ObstacleBuild(
            type=self.type,
            spawn=(spawn[0], spawn[1], spawn[2], spawn[3]),
            terrain_origin=(0.0, 0.0, 0.0),
            success_distance=float(p.get("success_distance", 4.0)),
            features=features,
            bounds=bounds,
            lane_width=6.0,
            metadata={"course_xml": str(course_xml), "geoms": len(list(course.iter("geom")))},
            assets=assets,
        )


COURSE = CourseObstacle()

OBSTACLES.register("course", COURSE)
