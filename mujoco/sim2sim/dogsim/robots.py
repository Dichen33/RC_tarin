"""Robot profiles: the only place that knows joint order and robot dimensions.

A profile is deliberately data-only.  The runtime asks the profile for names,
gains and observation dimensions; it never branches on "Go2W" or "MiniDog".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotProfile:
    name: str
    label: str

    # Joint and body contract (policy order, not necessarily XML order).
    joint_names: tuple[str, ...]
    position_joint_names: tuple[str, ...]
    velocity_joint_names: tuple[str, ...]
    default_angles: tuple[float, ...]
    base_body: str
    foot_bodies: tuple[str, ...]
    thigh_bodies: tuple[str, ...] = ()
    calf_bodies: tuple[str, ...] = ()

    # Actuator contract.
    position_actuator_suffix: str = "_position"
    velocity_actuator_suffix: str = "_velocity"
    kp: float = 45.0
    kv: float = 1.2
    velocity_kv: float | None = None
    effort: float = 20.0
    position_action_scale: float = 0.25
    velocity_action_scale: float = 8.0

    # Geometry / observation contract.
    stand_height: float = 0.45
    wheel_radius: float = 0.0
    history_length: int = 6
    observation_term_dims: tuple[int, ...] = (3, 3, 3, 2, 12, 12, 12)
    observation_term_names: tuple[str, ...] = (
        "base_ang_vel",
        "projected_gravity",
        "command",
        "gait_phase",
        "joint_pos",
        "joint_vel",
        "last_action",
    )
    scan_dim: int = 403
    scan_size: tuple[float, float] = (2.4, 1.0)
    scan_resolution: float = 0.08
    scan_size_by_dim: dict[int, tuple[float, float]] = field(
        default_factory=lambda: {216: (1.4, 0.9), 403: (2.4, 1.0)}
    )

    # Gait clock used by the observation.
    gait_base_frequency: float = 1.8
    gait_max_frequency: float = 2.8
    gait_command_threshold: float = 0.03

    # Height scan is defined relative to the base body.  The wheel/leg scan
    # exclusion is handled by putting terrain on geom group 0 and the robot on
    # groups 1/3, matching the training scenes.
    scan_include_geom_groups: tuple[int, ...] = (0,)

    # Suggested XML locations, relative to the workspace root.  The config
    # may override the path; these are only used for discovery and messages.
    xml_hints: tuple[str, ...] = ()

    @property
    def action_dim(self) -> int:
        return len(self.position_joint_names) + len(self.velocity_joint_names)

    @property
    def frame_dim(self) -> int:
        return int(sum(self.observation_term_dims))

    @property
    def proprio_dim(self) -> int:
        return self.frame_dim * self.history_length

    @property
    def all_contact_bodies(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.foot_bodies, *self.calf_bodies, *self.thigh_bodies, self.base_body)))

    def default_joint_vector(self) -> tuple[float, ...]:
        if len(self.default_angles) != len(self.joint_names):
            raise ValueError(
                f"{self.name}: {len(self.default_angles)} default angles for "
                f"{len(self.joint_names)} joints"
            )
        return self.default_angles


GO2W_JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
)

GO2W = RobotProfile(
    name="go2w",
    label="Unitree Go2W (16 DoF wheel-legged)",
    joint_names=GO2W_JOINT_NAMES,
    position_joint_names=GO2W_JOINT_NAMES[:12],
    velocity_joint_names=GO2W_JOINT_NAMES[12:],
    default_angles=(0.0, 0.8, -1.5) * 4 + (0.0, 0.0, 0.0, 0.0),
    base_body="base",
    foot_bodies=("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
    thigh_bodies=("FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"),
    calf_bodies=("FR_calf", "FL_calf", "RR_calf", "RL_calf"),
    kp=25.0,
    kv=0.5,
    velocity_kv=0.15,
    effort=23.5,
    position_action_scale=0.25,
    velocity_action_scale=8.0,
    stand_height=0.45,
    wheel_radius=0.086,
    history_length=6,
    observation_term_dims=(3, 3, 3, 2, 16, 16, 16),
    scan_dim=403,
    scan_size=(2.4, 1.0),
    xml_hints=(
        "HighTorque-MiniDog/mujoco/mjlab/go2w/go2w_description.xml",
        "../mjlab/go2w/go2w_description.xml",
        "go2w/go2w_description.xml",
    ),
)

MINIDOG_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

MINIDOG = RobotProfile(
    name="minidog",
    label="HighTorque MiniDog (12 DoF)",
    joint_names=MINIDOG_JOINT_NAMES,
    position_joint_names=MINIDOG_JOINT_NAMES,
    velocity_joint_names=(),
    default_angles=(0.0, 0.4838, -0.9461) * 4,
    base_body="base_link",
    foot_bodies=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
    thigh_bodies=("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"),
    calf_bodies=("FL_calf", "FR_calf", "RL_calf", "RR_calf"),
    kp=45.0,
    kv=1.2,
    velocity_kv=None,
    effort=20.0,
    position_action_scale=0.25,
    velocity_action_scale=0.0,
    stand_height=0.262,
    wheel_radius=0.0,
    history_length=6,
    observation_term_dims=(3, 3, 3, 2, 12, 12, 12),
    scan_dim=403,
    scan_size=(2.4, 1.0),
    xml_hints=(
        "HighTorque-MiniDog/mujoco/htdw_4438/htdw_4438.xml",
        "../htdw_4438/htdw_4438.xml",
        "htdw_4438/htdw_4438.xml",
    ),
)

PROFILES: dict[str, RobotProfile] = {GO2W.name: GO2W, MINIDOG.name: MINIDOG}


def get_profile(name: str) -> RobotProfile:
    key = name.strip().lower()
    if key not in PROFILES:
        raise KeyError(f"Unknown robot profile '{name}'. Available: {', '.join(sorted(PROFILES))}")
    return PROFILES[key]


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(PROFILES))
