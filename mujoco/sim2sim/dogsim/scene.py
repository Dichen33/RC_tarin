"""MJCF composition: robot + one obstacle brick + validated physics options."""

from __future__ import annotations

import copy
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .config import ObstacleConfig, RunConfig, SimulationConfig
from .obstacles import BuildContext, ObstacleBuild, get_obstacle
from .robots import RobotProfile


@dataclass
class JointMap:
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    qvel_ids: np.ndarray
    position_actuator_ids: np.ndarray
    velocity_actuator_ids: np.ndarray
    position_action_indices: np.ndarray
    velocity_action_indices: np.ndarray


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    build: ObstacleBuild
    output_xml: Path
    robot: RobotProfile
    joints: JointMap
    base_body_id: int
    robot_body_ids: set[int]
    robot_geom_ids: set[int]
    terrain_geom_ids: set[int]
    foot_body_ids: np.ndarray
    foot_geom_ids: set[int]
    feature_geoms: dict[str, set[int]] = field(default_factory=dict)

    def geom_id(self, name: str) -> int:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise KeyError(f"Geom '{name}' not found in {self.output_xml}")
        return int(gid)

    def body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"Body '{name}' not found in {self.output_xml}")
        return int(bid)


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == int(ancestor_id):
            return True
        current = int(model.body_parentid[current])
    return False


def _joint_range(robot: ET.Element, joint_name: str) -> tuple[float, float]:
    for joint in robot.iter("joint"):
        if joint.get("name") != joint_name:
            continue
        raw = joint.get("range")
        if raw:
            low, high = (float(v) for v in raw.split())
            if low > high:
                low, high = high, low
            return low, high
        break
    if "hip" in joint_name:
        return -1.05, 1.05
    if "thigh" in joint_name:
        return -1.57, 3.49
    if "calf" in joint_name:
        return -2.70, 0.0
    return -3.14, 3.14


def _replace_actuators(robot: ET.Element, profile: RobotProfile) -> None:
    actuator = robot.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(robot, "actuator")
    for child in list(actuator):
        actuator.remove(child)
    for joint in profile.position_joint_names:
        low, high = _joint_range(robot, joint)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint}{profile.position_actuator_suffix}",
                "joint": joint,
                "kp": f"{profile.kp:g}",
                "kv": f"{profile.kv:g}",
                "ctrlrange": f"{low:g} {high:g}",
                "ctrllimited": "true",
                "forcelimited": "true",
                "forcerange": f"{-profile.effort:g} {profile.effort:g}",
            },
        )
    for joint in profile.velocity_joint_names:
        kv = profile.kv if profile.velocity_kv is None else profile.velocity_kv
        ET.SubElement(
            actuator,
            "velocity",
            {
                "name": f"{joint}{profile.velocity_actuator_suffix}",
                "joint": joint,
                "kv": f"{kv:g}",
                "forcelimited": "true",
                "forcerange": f"{-profile.effort:g} {profile.effort:g}",
            },
        )


def _ensure_freejoint(robot: ET.Element) -> None:
    worldbody = robot.find("worldbody")
    if worldbody is None:
        raise ValueError("Robot XML has no <worldbody>")
    root = next((child for child in worldbody if child.tag == "body"), None)
    if root is None:
        raise ValueError("Robot XML has no root body")
    has_free = any(
        child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free")
        for child in root
    )
    if not has_free:
        ET.SubElement(root, "freejoint", {"name": f"{root.get('name', 'base')}_freejoint"})


def _prepare_robot_geoms(robot: ET.Element) -> None:
    """Keep robot geoms off the height-scan inclusion group.

    Training scenes put the robot on visual/collision groups 1/3 and terrain on
    group 0.  A robot geom accidentally left on group 0 would be seen by the
    downward height rays as a false obstacle.
    """

    for geom in robot.iter("geom"):
        if geom.get("group", "0") == "0":
            geom.set("group", "1")


def _set_compiler(robot: ET.Element, robot_xml: Path) -> None:
    compiler = robot.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(robot, "compiler")
    compiler.set("angle", "radian")
    meshdir = robot_xml.parent / "meshes"
    if meshdir.exists():
        compiler.set("meshdir", str(meshdir.resolve()).replace("\\", "/"))
    texturedir = robot_xml.parent / "textures"
    if texturedir.exists():
        compiler.set("texturedir", str(texturedir.resolve()).replace("\\", "/"))


def _add_light(robot: ET.Element) -> None:
    worldbody = robot.find("worldbody")
    assert worldbody is not None
    if not any(child.tag == "light" for child in worldbody):
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "key_light",
                "pos": "0 0 6",
                "dir": "0 0 -1",
                "directional": "true",
                "diffuse": "0.7 0.7 0.7",
                "ambient": "0.25 0.25 0.25",
            },
        )


def _add_statistic(robot: ET.Element, build: ObstacleBuild) -> None:
    old = robot.find("statistic")
    if old is not None:
        robot.remove(old)
    x0, x1, y0, y1 = build.bounds
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0)
    extent = max(x1 - x0, y1 - y0, 4.0)
    ET.SubElement(
        robot,
        "statistic",
        {"center": " ".join(f"{v:g}" for v in center), "extent": f"{extent:g}"},
    )


def _apply_physics_options(model: mujoco.MjModel, cfg: SimulationConfig) -> None:
    integrators = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    cones = {
        "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
        "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
    }
    solvers = {
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "pgs": mujoco.mjtSolver.mjSOL_PGS,
    }
    model.opt.timestep = float(cfg.physics_dt)
    model.opt.integrator = integrators[cfg.integrator]
    model.opt.cone = cones[cfg.cone]
    model.opt.impratio = float(cfg.impratio)
    model.opt.solver = solvers[cfg.solver]
    model.opt.iterations = int(cfg.iterations)
    model.opt.tolerance = float(cfg.tolerance)
    model.opt.ls_iterations = int(cfg.ls_iterations)
    model.opt.ls_tolerance = float(cfg.ls_tolerance)
    model.opt.ccd_iterations = int(cfg.ccd_iterations)
    model.opt.gravity[:] = np.asarray(cfg.gravity, dtype=np.float64)


def _build_joint_map(model: mujoco.MjModel, profile: RobotProfile) -> JointMap:
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in profile.joint_names],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        missing = [name for name, jid in zip(profile.joint_names, joint_ids) if jid < 0]
        raise ValueError(f"Robot XML is missing joints: {missing}")
    qpos_ids = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
    qvel_ids = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
    position_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}{profile.position_actuator_suffix}"
            )
            for name in profile.position_joint_names
        ],
        dtype=np.int32,
    )
    velocity_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}{profile.velocity_actuator_suffix}"
            )
            for name in profile.velocity_joint_names
        ],
        dtype=np.int32,
    )
    if np.any(position_ids < 0):
        raise ValueError("Missing position actuators; check the robot XML and actuator suffix")
    if velocity_ids.size and np.any(velocity_ids < 0):
        raise ValueError("Missing velocity actuators; check the robot XML and actuator suffix")
    index_of = {name: i for i, name in enumerate(profile.joint_names)}
    position_indices = np.asarray([index_of[name] for name in profile.position_joint_names], dtype=np.int32)
    velocity_indices = np.asarray([index_of[name] for name in profile.velocity_joint_names], dtype=np.int32)
    if len(position_indices) + len(velocity_indices) != len(profile.joint_names):
        raise ValueError("Position and velocity joint lists do not partition joint_names")
    return JointMap(
        joint_ids=joint_ids,
        qpos_ids=qpos_ids,
        qvel_ids=qvel_ids,
        position_actuator_ids=position_ids,
        velocity_actuator_ids=velocity_ids,
        position_action_indices=position_indices,
        velocity_action_indices=velocity_indices,
    )


def compose_scene(cfg: RunConfig, output_xml: Path | None = None) -> Scene:
    """Compose robot + obstacle brick and return a loaded, validated scene."""

    robot_xml = cfg.robot_xml
    robot = ET.parse(robot_xml).getroot()
    robot.set("model", f"sim2sim_{cfg.robot.name}_{cfg.obstacle.type}")
    _set_compiler(robot, robot_xml)
    _ensure_freejoint(robot)
    _prepare_robot_geoms(robot)
    _replace_actuators(robot, cfg.robot)
    _add_light(robot)

    worldbody = robot.find("worldbody")
    if worldbody is None:
        raise ValueError("Robot XML has no <worldbody>")
    module = get_obstacle(cfg.obstacle.type)
    rng = np.random.default_rng(cfg.simulation.randomization.seed + 17)
    ctx = BuildContext(
        worldbody=worldbody,
        tag=f"lane_{cfg.obstacle.type}",
        origin=tuple(float(v) for v in cfg.obstacle.origin),  # type: ignore[arg-type]
        yaw=float(cfg.obstacle.yaw),
        difficulty=float(cfg.obstacle.difficulty),
        rng=rng,
        robot_name=cfg.robot.name,
        stand_height=float(cfg.robot.stand_height),
        wheel_radius=float(cfg.robot.wheel_radius),
        params=dict(cfg.obstacle.params),
        base_dir=None if cfg.source_config is None else cfg.source_config.parent,
    )
    build = module.build(ctx)
    if build.assets:
        robot_asset = robot.find("asset")
        if robot_asset is None:
            robot_asset = ET.SubElement(robot, "asset")
        for asset in build.assets:
            robot_asset.append(copy.deepcopy(asset))
    _add_statistic(robot, build)

    output_xml = Path(output_xml) if output_xml is not None else cfg.output.scene_xml
    ET.indent(robot, space="  ")
    tree = ET.ElementTree(robot)
    try:
        output_xml.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    except PermissionError:
        # The original sim2sim/generated can be ACL-locked in some sessions.
        # Fall back to the OS temp directory so a viewer/UI still starts; the
        # caller can pass --scene to choose an explicit writable location.
        fallback = Path(tempfile.gettempdir()) / f"sim2sim_scene_{os.getpid()}.xml"
        print(f"[Scene] cannot write {output_xml}; using fallback {fallback}")
        tree.write(fallback, encoding="utf-8", xml_declaration=True)
        output_xml = fallback

    model = mujoco.MjModel.from_xml_path(str(output_xml))
    _apply_physics_options(model, cfg.simulation)
    data = mujoco.MjData(model)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.robot.base_body)
    if base_body_id < 0:
        raise ValueError(f"Robot body '{cfg.robot.base_body}' not found")
    robot_body_ids = {
        body_id
        for body_id in range(1, model.nbody)
        if _is_descendant(model, body_id, base_body_id)
    }
    robot_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in robot_body_ids
    }
    terrain_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_group[geom_id]) == 0 and geom_id not in robot_geom_ids
    }
    if not terrain_geom_ids:
        raise ValueError("Scene has no terrain geoms on group 0; check obstacle composition")
    foot_body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in cfg.robot.foot_bodies],
        dtype=np.int32,
    )
    if np.any(foot_body_ids < 0):
        raise ValueError("Robot XML is missing one or more foot bodies")
    foot_geom_ids = {
        geom_id
        for geom_id in robot_geom_ids
        if int(model.geom_bodyid[geom_id]) in set(int(v) for v in foot_body_ids)
        or (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").endswith("_foot_collision")
    }
    feature_geoms: dict[str, set[int]] = {}
    for feature in build.features:
        ids: set[int] = set()
        for name in feature.geoms:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                ids.add(int(gid))
        feature_geoms[feature.name] = ids

    joints = _build_joint_map(model, cfg.robot)
    scene = Scene(
        model=model,
        data=data,
        build=build,
        output_xml=output_xml,
        robot=cfg.robot,
        joints=joints,
        base_body_id=int(base_body_id),
        robot_body_ids=robot_body_ids,
        robot_geom_ids=robot_geom_ids,
        terrain_geom_ids=terrain_geom_ids,
        foot_body_ids=foot_body_ids,
        foot_geom_ids=foot_geom_ids,
        feature_geoms=feature_geoms,
    )

    # Validate the spawn pose before the runner starts.  This catches a lane
    # whose origin or platform height was configured incorrectly.
    spawn = scene.build.spawn
    data.qpos[:7] = (spawn[0], spawn[1], spawn[2], math.cos(spawn[3] / 2.0), 0.0, 0.0, math.sin(spawn[3] / 2.0))
    data.qpos[joints.qpos_ids] = np.asarray(cfg.robot.default_angles, dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise ValueError("Spawn pose produced non-finite physics state")
    return scene
