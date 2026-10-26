from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


JOINTS = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
JOINT_RANGES = {
    "hip": (-1.05, 1.05),
    "thigh": (-1.57, 3.49),
    "calf": (-2.7, 0.0),
}

# Keep the ROBOCON start zone at (0, 0) unobstructed.  The added training
# lane sits beyond the competition obstacles on the north side of the map.
PLUM_LANE_Y = 3.35
GAP_LANE_Y = -3.35
GAP_START_X = -0.55
GAP_WIDTH = 0.42
GAP_END_X = GAP_START_X + GAP_WIDTH
GAP_DEPTH = 0.22


def _geom(body: ET.Element, name: str, geom_type: str, pos: tuple[float, ...], size: tuple[float, ...], rgba: tuple[float, ...], **attributes) -> None:
    values = {
        "name": name,
        "type": geom_type,
        "pos": " ".join(f"{value:g}" for value in pos),
        "size": " ".join(f"{value:g}" for value in size),
        "rgba": " ".join(f"{value:g}" for value in rgba),
        "class": "course",
    }
    values.update(attributes)
    ET.SubElement(body, "geom", values)


def _add_minidog_obstacle_lanes(worldbody: ET.Element) -> None:
    """Add the MiniDog training obstacles beside the reference 27rc course.

    The 27rc XML contains the competition-style beam, wall, stones, slalom,
    tire, clearance and ramp lanes. These additional lanes preserve the
    obstacles used by MiniDog training, so the same UI can validate both.
    """
    # A shallow deterministic rough patch under the plum-blossom lane.
    rough = ET.SubElement(
        worldbody, "body", name="minidog_rough_plum_lane", pos=f"0 {PLUM_LANE_Y:g} 0"
    )
    rough_heights = (0.02, 0.04, 0.01, 0.05, 0.03, 0.06, 0.02, 0.04, 0.03, 0.01, 0.05, 0.02, 0.04, 0.03)
    index = 0
    for x in (-2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4):
        for y in (-0.4, 0.0, 0.4):
            height = rough_heights[index % len(rough_heights)]
            index += 1
            _geom(
                rough,
                f"minidog_rough_{index:03d}",
                "box",
                (x, y, -0.5 + height / 2.0),
                (0.2, 0.2, 0.5 + height / 2.0),
                (0.46 + height, 0.46 + height, 0.46 + height, 1.0),
            )

    # Same dimensions as RoughPlumBlossomTerrainCfg at medium difficulty.
    plum = ET.SubElement(
        worldbody, "body", name="minidog_plum_blossom", pos=f"0 {PLUM_LANE_Y:g} 0"
    )
    for name, x in (("entry", -2.2), ("exit", 2.2)):
        _geom(plum, f"minidog_plum_{name}_platform", "box", (x, 0.0, 0.10), (0.4, 0.4, 0.10), (0.35, 0.62, 0.20, 1.0))
    block_size = 0.29
    block_height = 0.15
    for block_index, (x, y) in enumerate((
        (-1.6, 0.2), (-1.0, 0.2), (-0.4, 0.2),
        (-1.3, -0.2), (-0.7, -0.2), (-0.1, -0.2),
    )):
        _geom(
            plum,
            f"minidog_plum_block_{block_index}",
            "box",
            (x, y, block_height / 2.0),
            (block_size / 2.0, block_size / 2.0, block_height / 2.0),
            (0.35, 0.62, 0.20, 1.0),
        )

    # Independent flat-ground high-wall lane, matching the training wall
    # geometry rather than combining walls with the plum-blossom lane.
    walls = ET.SubElement(worldbody, "body", name="minidog_flat_high_wall", pos="0 0.85 0")
    for index, x in enumerate((-1.6, 0.0, 1.6)):
        _geom(walls, f"minidog_high_wall_{index}", "box", (x, 0.0, 0.09), (0.025, 0.36, 0.09), (0.72, 0.30, 0.16, 1.0))

    # Independent low-bar lane using the same 0.24-0.34 m clearance range.
    bars = ET.SubElement(worldbody, "body", name="minidog_low_bar", pos="0 -0.85 0")
    for index, x in enumerate((-2.0, -1.2, -0.4, 0.4, 1.2, 2.0)):
        bar_z = 0.305
        _geom(
            bars,
            f"minidog_low_bar_{index}",
            "cylinder",
            (x, 0.0, bar_z),
            (0.025, 0.425),
            (0.18, 0.34, 0.72, 1.0),
            quat="0.70710678 0.70710678 0 0",
        )
        for post_index, y in enumerate((-0.425, 0.425)):
            _geom(
                bars,
                f"minidog_low_bar_{index}_post_{post_index}",
                "cylinder",
                (x, y, bar_z / 2.0),
                (0.025, bar_z / 2.0),
                (0.18, 0.34, 0.72, 1.0),
            )

    gap = ET.SubElement(worldbody, "body", name="minidog_flat_gap", pos=f"0 {GAP_LANE_Y:g} 0")
    for name, x0, x1 in (("approach", -3.0, GAP_START_X), ("landing", GAP_END_X, 4.2)):
        _geom(
            gap,
            f"minidog_gap_{name}",
            "box",
            ((x0 + x1) / 2.0, 0.0, -0.05),
            ((x1 - x0) / 2.0, 1.0, 0.05),
            (0.52, 0.52, 0.52, 1.0),
        )
    _geom(
        gap,
        "minidog_gap_bottom",
        "box",
        ((GAP_START_X + GAP_END_X) / 2.0, 0.0, -GAP_DEPTH - 0.04),
        ((GAP_END_X - GAP_START_X) / 2.0, 1.0, 0.04),
        (0.28, 0.28, 0.28, 1.0),
    )


def _radianize_course(node: ET.Element) -> None:
    for child in node.iter():
        if "euler" in child.attrib:
            values = [float(v) for v in child.attrib["euler"].split()]
            child.set("euler", " ".join(f"{math.radians(v):.12g}" for v in values))
        if "axisangle" in child.attrib:
            values = [float(v) for v in child.attrib["axisangle"].split()]
            values[-1] = math.radians(values[-1])
            child.set("axisangle", " ".join(f"{v:.12g}" for v in values))


def _resolve_course_assets(course: ET.Element, course_xml: Path) -> None:
    """Make copied course assets independent of the generated XML location.

    ROBOCON2027.xml contains inline meshes and references terrain.png relative
    to its original directory.  The composed scene is written to generated/,
    so relative paths would otherwise resolve against the wrong directory.
    """
    compiler = course.find("compiler")
    meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    texturedir = Path(compiler.get("texturedir", "")) if compiler is not None else Path()
    asset = course.find("asset")
    if asset is None:
        return

    for node in list(asset):
        file_name = node.get("file")
        if not file_name or Path(file_name).is_absolute():
            continue
        base = course_xml.parent
        if node.tag == "mesh":
            base = base / meshdir
        elif node.tag == "texture":
            base = base / texturedir
        candidate = (base / file_name).resolve()
        if candidate.exists():
            node.set("file", candidate.as_posix())
            continue

        # The supplied map declares an unused hfield asset, while its actual
        # competition obstacles are direct box/mesh geoms.  Do not make the
        # whole scene depend on a missing optional image.
        if node.tag == "hfield" and not any(
            geom.get("hfield") == node.get("name")
            for geom in course.iter("geom")
        ):
            asset.remove(node)
            continue
        raise FileNotFoundError(
            f"Course asset not found: {candidate} (referenced by {course_xml})"
        )


def _replace_actuators(
    robot: ET.Element,
    position_joints: tuple[str, ...] = JOINTS,
    velocity_joints: tuple[str, ...] = (),
    kp: float = 45.0,
    kv: float = 1.2,
    velocity_kv: float | None = None,
    effort: float = 20.0,
) -> None:
    def joint_range(joint_name: str) -> tuple[float, float]:
        for joint in robot.iter("joint"):
            if joint.get("name") != joint_name:
                continue
            values = joint.get("range")
            if values:
                lower, upper = (float(value) for value in values.split())
                return lower, upper
            break
        if "hip" in joint_name:
            return JOINT_RANGES["hip"]
        if "thigh" in joint_name:
            return JOINT_RANGES["thigh"]
        return JOINT_RANGES["calf"]

    actuator = robot.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(robot, "actuator")
    for child in list(actuator):
        actuator.remove(child)
    for joint in position_joints:
        lo, hi = joint_range(joint)
        ET.SubElement(
            actuator,
            "position",
            name=f"{joint}_position",
            joint=joint,
            kp=f"{kp:g}",
            kv=f"{kv:g}",
            ctrlrange=f"{lo} {hi}",
            ctrllimited="true",
            forcelimited="true",
            forcerange=f"{-effort:g} {effort:g}",
        )
    for joint in velocity_joints:
        ET.SubElement(
            actuator,
            "velocity",
            name=f"{joint}_velocity",
            joint=joint,
            kv=f"{(kv if velocity_kv is None else velocity_kv):g}",
            forcelimited="true",
            forcerange=f"{-effort:g} {effort:g}",
        )


def compose_scene(
    robot_xml: Path,
    course_xml: Path,
    output_xml: Path,
    *,
    actuator_kp: float = 45.0,
    actuator_kv: float = 1.2,
    velocity_actuator_kv: float | None = None,
    actuator_effort: float = 20.0,
    position_joints: tuple[str, ...] = JOINTS,
    velocity_joints: tuple[str, ...] = (),
) -> Path:
    robot = ET.parse(robot_xml).getroot()
    course = ET.parse(course_xml).getroot()
    _radianize_course(course)
    _resolve_course_assets(course, course_xml)

    robot.set("model", "minidog_robocon_2027_sim2sim")
    robot_compiler = robot.find("compiler")
    if robot_compiler is None:
        robot_compiler = ET.SubElement(robot, "compiler")
    robot_compiler.set("angle", "radian")
    # The generated XML lives under sim2sim/generated, while the robot meshes
    # remain beside the original htdw_4438.xml.
    robot_compiler.set("meshdir", str((robot_xml.parent / "meshes").resolve()).replace("\\", "/"))

    robot_asset = robot.find("asset")
    course_asset = course.find("asset")
    if robot_asset is None:
        robot_asset = ET.SubElement(robot, "asset")
    if course_asset is not None:
        for asset in course_asset:
            robot_asset.append(copy.deepcopy(asset))

    robot_default = robot.find("default")
    course_default = course.find("default")
    if robot_default is None:
        robot_default = ET.SubElement(robot, "default")
    if course_default is None:
        # ROBOCON2027.xml uses explicit geom attributes and has no default
        # block, but its copied geoms still receive class="course" below.
        course_default = ET.Element("default")
    else:
        course_default = copy.deepcopy(course_default)
    course_default.set("class", "course")
    robot_default.append(course_default)

    robot_world = robot.find("worldbody")
    course_world = course.find("worldbody")
    if robot_world is None or course_world is None:
        raise ValueError("Robot and course XML must both contain worldbody")
    root_body = next((child for child in robot_world if child.tag == "body"), None)
    if root_body is not None:
        has_free = any(child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free") for child in root_body)
        if not has_free:
            ET.SubElement(root_body, "freejoint", {"name": f"{root_body.get('name', 'base')}_freejoint"})
    for element in course_world:
        element = copy.deepcopy(element)
        for geom in element.iter("geom"):
            geom.set("class", "course")
        robot_world.append(element)
    _add_minidog_obstacle_lanes(robot_world)

    for tag in ("statistic", "visual"):
        old = robot.find(tag)
        if old is not None:
            robot.remove(old)
        source = course.find(tag)
        if source is not None:
            robot.append(copy.deepcopy(source))

    _replace_actuators(
        robot,
        position_joints=position_joints,
        velocity_joints=velocity_joints,
        kp=actuator_kp,
        kv=actuator_kv,
        velocity_kv=velocity_actuator_kv,
        effort=actuator_effort,
    )
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(output_xml, encoding="utf-8", xml_declaration=True)
    # Loading here makes XML and asset errors fail before a viewer is opened.
    mujoco.MjModel.from_xml_path(str(output_xml))
    return output_xml
