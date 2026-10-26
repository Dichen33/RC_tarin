from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


MJLAB_ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = MJLAB_ROOT / "go2w" / "go2w_description.xml"

# This is the order used by Unitree's Go2W deployment code and is the policy
# contract. The XML's natural order is FL, FR, RL, RR, so actions are created
# as one exact-name action term per joint below.
GO2W_JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
)
GO2W_LEG_JOINT_NAMES = GO2W_JOINT_NAMES[:12]
GO2W_WHEEL_JOINT_NAMES = GO2W_JOINT_NAMES[12:]
GO2W_FOOT_BODY_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
GO2W_WHEEL_GEOM_NAMES = tuple(
    f"{body_name}_geom_{index}"
    for body_name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    for index in range(2)
)
GO2W_LEG_BODY_NAMES = (
    "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh",
    "FR_calf", "FL_calf", "RR_calf", "RL_calf",
)

STAND_HEIGHT = 0.45
CRAWL_HEIGHT = 0.36
LEG_ACTION_SCALE = 0.25
WHEEL_ACTION_SCALE = 8.0
# Measured from the collision mesh in ``go2w/meshes/dae/left_wheel.stl``
# (bounding box x/z in [-0.086, 0.086]).  The foot body origin is the wheel
# axle centre, so a grounded wheel has its axle exactly WHEEL_RADIUS above the
# surface it stands on.  Every reward that compares a foot height with a block
# or wall top must subtract this value.
WHEEL_RADIUS = 0.086
# Nominal wheel-axle positions in the base frame, measured from the model:
# wheelbase 0.387 m (0.186 front / -0.200 rear), track 0.284 m (+/-0.142).  The
# course geometry is compared against these offsets (16 mm lateral margin against
# the 0.20 m gutter), so they must have a single definition.
WHEEL_OFFSETS = {
    "FL": (0.186, 0.142),
    "FR": (0.186, -0.142),
    "RL": (-0.200, 0.142),
    "RR": (-0.200, -0.142),
}

STAND_JOINT_POS = {
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
    ".*_foot_joint": 0.0,
}


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    # The source XML leaves mesh geoms unnamed.  Give every unnamed geom a
    # stable body-local name so contact parameters can be randomized by body
    # without relying on fragile global geom indices.  This also keeps the
    # wheel geom selection explicit for sim2sim traction randomization.
    geom_counts: dict[str, int] = {}
    for geom in spec.geoms:
        if not geom.name:
            body_name = geom.parent.name
            index = geom_counts.get(body_name, 0)
            geom.name = f"{body_name}_geom_{index}"
            geom_counts[body_name] = index + 1
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)
    if not any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in spec.joints):
        spec.body("base").add_joint(name="base_freejoint", type=mujoco.mjtJoint.mjJNT_FREE)
    return spec


LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=GO2W_LEG_JOINT_NAMES,
    stiffness=25.0,
    damping=0.5,
    effort_limit=23.5,
)
WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
    target_names_expr=GO2W_WHEEL_JOINT_NAMES,
    damping=0.15,
    effort_limit=23.5,
)
ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(LEG_ACTUATOR_CFG, WHEEL_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.95,
)


def get_robot_cfg(crawl: bool = False) -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, CRAWL_HEIGHT if crawl else STAND_HEIGHT),
            joint_pos=STAND_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )

