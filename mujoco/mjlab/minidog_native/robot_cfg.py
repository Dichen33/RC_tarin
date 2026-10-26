from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


MJLAB_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_ROOT = MJLAB_ROOT.parent
ROBOT_XML = MUJOCO_ROOT / "htdw_4438" / "htdw_4438.xml"

LEG_JOINT_PATTERNS = (
    ".*_hip_joint",
    ".*_thigh_joint",
    ".*_calf_joint",
)

FOOT_BODY_PATTERNS = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)

FOOT_GEOM_PATTERNS = tuple(f"{leg}_foot_collision" for leg in ("FL", "FR", "RL", "RR"))
THIGH_GEOM_PATTERNS = tuple(f"{leg}_thigh_collision" for leg in ("FL", "FR", "RL", "RR"))
CALF_GEOM_PATTERNS = tuple(f"{leg}_calf_collision" for leg in ("FL", "FR", "RL", "RR"))
BASE_GEOM_PATTERNS = ("base_collision",)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    # The source XML keyframe contains the original 12 motor controls.  The
    # native MJLab entity adds its own actuators, so keep the pose but remove
    # the stale control vector before compiling the spec.
    for key in spec.keys:
        key.ctrl = []
    return spec


STIFFNESS = 45.0
DAMPING = 1.2
EFFORT_LIMIT = 20.0
ACTION_SCALE = 0.25

STAND_HEIGHT = 0.235
CRAWL_HEIGHT = 0.175

STAND_JOINT_POS = {
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.4838,
    ".*_calf_joint": -0.9461,
}

CRAWL_JOINT_POS = {
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 1.15,
    ".*_calf_joint": -1.95,
}

ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=LEG_JOINT_PATTERNS,
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
)

COLLISION_CFG = CollisionCfg(
    geom_names_expr=BASE_GEOM_PATTERNS + FOOT_GEOM_PATTERNS + THIGH_GEOM_PATTERNS + CALF_GEOM_PATTERNS,
    contype=1,
    conaffinity=1,
    condim={".*_foot_collision": 3, ".*_(thigh|calf)_collision": 3, "base_collision": 3},
    priority={".*_foot_collision": 1, ".*_(thigh|calf)_collision": 1, "base_collision": 1},
    friction={
        ".*_foot_collision": (1.0, 0.02, 0.001),
        ".*_(thigh|calf)_collision": (0.8, 0.02, 0.001),
        "base_collision": (0.8, 0.02, 0.001),
    },
)

ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.95,
)


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, STAND_HEIGHT),
            joint_pos=STAND_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        collisions=(COLLISION_CFG,),
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )


def get_robot_crawl_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, CRAWL_HEIGHT),
            joint_pos=CRAWL_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        collisions=(COLLISION_CFG,),
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )
