from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimConfig:
    """Configuration shared by policy reconstruction and MuJoCo stepping."""

    robot_xml: Path
    course_xml: Path
    physics_dt: float
    control_decimation: int
    action_scale: float
    action_clip: float
    actuator_kp: float
    actuator_kv: float
    velocity_actuator_kv: float | None
    actuator_effort: float
    default_angles: tuple[float, ...]
    lin_vel_scale: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    command_scale: tuple[float, ...]
    joint_names: tuple[str, ...]
    actuator_suffix: str
    history_length: int
    observation_term_dims: tuple[int, ...]
    scan_resolution: float
    scan_size: tuple[float, float] | None
    scan_stride: int
    scan_update_interval: int
    command: tuple[float, float, float]
    spawn: tuple[float, float, float, float]
    duration: float
    max_command: tuple[float, float, float]
    position_joint_names: tuple[str, ...] = ()
    velocity_joint_names: tuple[str, ...] = ()
    position_actuator_suffix: str = "_position"
    velocity_actuator_suffix: str = "_velocity"
    base_body_name: str = "base_link"
    foot_body_names: tuple[str, ...] = ()
    position_action_scale: float | None = None
    velocity_action_scale: float | None = None
    scan_dim: int | None = None
    randomization_enabled: bool = True
    random_seed: int = 42
    ground_friction_range: tuple[float, float] = (0.75, 1.25)
    wheel_friction_range: tuple[float, float] = (0.70, 1.30)
    joint_damping_scale_range: tuple[float, float] = (0.85, 1.15)
    joint_friction_scale_range: tuple[float, float] = (0.80, 1.20)
    actuator_gain_scale_range: tuple[float, float] = (0.90, 1.10)
    mass_scale_range: tuple[float, float] = (0.95, 1.05)
    com_offset_range: float = 0.015
    motor_response_range: tuple[float, float] = (0.75, 1.0)
    action_delay_steps_range: tuple[int, int] = (0, 2)
    control_decimation_jitter: int = 1
    imu_noise_std: float = 0.015
    joint_pos_noise_std: float = 0.003
    joint_vel_noise_std: float = 0.03
    scan_noise_std: float = 0.015
    scan_dropout_prob: float = 0.02


def _path(value: Any, config_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _float_range(raw: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = raw.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"randomization.{key} must contain two values")
    low, high = float(value[0]), float(value[1])
    if low > high or low < 0.0:
        raise ValueError(f"randomization.{key} must be a non-negative ascending range")
    return low, high


def _int_range(raw: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
    value = raw.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"randomization.{key} must contain two values")
    low, high = int(value[0]), int(value[1])
    if low < 0 or low > high:
        raise ValueError(f"randomization.{key} must be a non-negative ascending range")
    return low, high


def load_sim_config(config_path: Path) -> SimConfig:
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    randomization = raw.get("randomization", {}) or {}
    return SimConfig(
        robot_xml=_path(raw["robot_xml"], config_path),
        course_xml=_path(raw["course_xml"], config_path),
        physics_dt=float(raw["physics_dt"]),
        control_decimation=int(raw["control_decimation"]),
        action_scale=float(raw["action_scale"]),
        action_clip=float(raw["action_clip"]),
        actuator_kp=float(raw["actuator_kp"]),
        actuator_kv=float(raw["actuator_kv"]),
        velocity_actuator_kv=(
            None
            if raw.get("velocity_actuator_kv") is None
            else float(raw["velocity_actuator_kv"])
        ),
        actuator_effort=float(raw["actuator_effort"]),
        default_angles=tuple(float(v) for v in raw["default_angles"]),
        lin_vel_scale=float(raw["lin_vel_scale"]),
        ang_vel_scale=float(raw["ang_vel_scale"]),
        dof_pos_scale=float(raw["dof_pos_scale"]),
        dof_vel_scale=float(raw["dof_vel_scale"]),
        command_scale=tuple(float(v) for v in raw["command_scale"]),
        joint_names=tuple(str(v) for v in raw["joint_names"]),
        actuator_suffix=str(raw.get("actuator_suffix", raw.get("position_actuator_suffix", "_position"))),
        history_length=int(raw["history_length"]),
        observation_term_dims=tuple(int(v) for v in raw["observation_term_dims"]),
        scan_resolution=float(raw["scan_resolution"]),
        scan_size=None if raw.get("scan_size") is None else tuple(float(v) for v in raw["scan_size"]),
        scan_stride=max(1, int(raw.get("scan_stride", 1))),
        scan_update_interval=max(1, int(raw.get("scan_update_interval", 1))),
        command=tuple(float(v) for v in raw["command"]),
        spawn=tuple(float(v) for v in raw["spawn"]),
        duration=float(raw["duration"]),
        max_command=tuple(float(v) for v in raw["max_command"]),
        position_joint_names=tuple(str(v) for v in raw.get("position_joint_names", raw["joint_names"])),
        velocity_joint_names=tuple(str(v) for v in raw.get("velocity_joint_names", ())),
        position_actuator_suffix=str(raw.get("position_actuator_suffix", raw.get("actuator_suffix", "_position"))),
        velocity_actuator_suffix=str(raw.get("velocity_actuator_suffix", "_velocity")),
        base_body_name=str(raw.get("base_body_name", "base_link")),
        foot_body_names=tuple(str(v) for v in raw.get("foot_body_names", ())),
        position_action_scale=None if raw.get("position_action_scale") is None else float(raw["position_action_scale"]),
        velocity_action_scale=None if raw.get("velocity_action_scale") is None else float(raw["velocity_action_scale"]),
        scan_dim=None if raw.get("scan_dim") is None else int(raw["scan_dim"]),
        randomization_enabled=bool(randomization.get("enabled", True)),
        random_seed=int(randomization.get("seed", 42)),
        ground_friction_range=_float_range(randomization, "ground_friction_range", (0.75, 1.25)),
        wheel_friction_range=_float_range(randomization, "wheel_friction_range", (0.70, 1.30)),
        joint_damping_scale_range=_float_range(randomization, "joint_damping_scale_range", (0.85, 1.15)),
        joint_friction_scale_range=_float_range(randomization, "joint_friction_scale_range", (0.80, 1.20)),
        actuator_gain_scale_range=_float_range(randomization, "actuator_gain_scale_range", (0.90, 1.10)),
        mass_scale_range=_float_range(randomization, "mass_scale_range", (0.95, 1.05)),
        com_offset_range=max(0.0, float(randomization.get("com_offset_range", 0.015))),
        motor_response_range=_float_range(randomization, "motor_response_range", (0.75, 1.0)),
        action_delay_steps_range=_int_range(randomization, "action_delay_steps_range", (0, 2)),
        control_decimation_jitter=max(0, int(randomization.get("control_decimation_jitter", 1))),
        imu_noise_std=max(0.0, float(randomization.get("imu_noise_std", 0.015))),
        joint_pos_noise_std=max(0.0, float(randomization.get("joint_pos_noise_std", 0.003))),
        joint_vel_noise_std=max(0.0, float(randomization.get("joint_vel_noise_std", 0.03))),
        scan_noise_std=max(0.0, float(randomization.get("scan_noise_std", 0.015))),
        scan_dropout_prob=min(max(float(randomization.get("scan_dropout_prob", 0.02)), 0.0), 1.0),
    )
