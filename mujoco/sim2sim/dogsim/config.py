"""Configuration loading, defaults and validation.

The runtime is configuration driven.  A YAML file selects a robot profile, a
policy checkpoint, one obstacle module and one evaluator.  Every value has a
documented default, and every invalid combination fails at load time with a
message that names the field.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .robots import RobotProfile, get_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_path(value: str | Path | None, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates: list[Path] = []
    if base is not None:
        candidates.append((base / path).resolve())
    candidates.append((PROJECT_ROOT / path).resolve())
    candidates.append((Path.cwd() / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path.resolve()


def discover_checkpoint(configured: str | Path | None, base: Path | None) -> Path | None:
    """Resolve a checkpoint, falling back to ancestors by filename.

    This keeps ``configs/go2w.yaml`` usable after the project is copied into
    the original ``sim2sim`` directory, where the relative checkpoint path
    points at a different level.
    """

    if not configured:
        return None
    resolved = resolve_path(configured, base)
    if resolved is not None and resolved.exists():
        return resolved
    name = Path(str(configured)).name
    for parent in (
        PROJECT_ROOT,
        PROJECT_ROOT.parent,
        PROJECT_ROOT.parent.parent,
        PROJECT_ROOT.parent.parent.parent,
    ):
        candidate = parent / name
        if candidate.exists():
            return candidate
    return resolved


def discover_robot_xml(profile: RobotProfile, configured: str | Path | None, base: Path | None) -> Path:
    if configured:
        resolved = resolve_path(configured, base)
        if resolved is not None and resolved.exists():
            return resolved
        # A configured path that does not exist is an error, but try the
        # profile hints first so a copied project keeps working.
    for hint in profile.xml_hints:
        candidate = resolve_path(hint, PROJECT_ROOT)
        if candidate is not None and candidate.exists():
            return candidate
        candidate = resolve_path(hint, PROJECT_ROOT.parent)
        if candidate is not None and candidate.exists():
            return candidate
    if configured:
        resolved = resolve_path(configured, base)
        assert resolved is not None
        raise FileNotFoundError(
            f"Robot XML not found: {resolved}. Set robot.xml or one of the profile hints "
            f"{profile.xml_hints}."
        )
    raise FileNotFoundError(
        f"No robot.xml configured and no profile hint exists for {profile.name}: {profile.xml_hints}"
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScanConfig:
    enabled: bool = True
    update_hz: float = 10.0
    noise_std: float = 0.03
    dropout_prob: float = 0.04
    stride: int = 1
    max_distance: float = 3.0
    clip_min: float = -1.0
    clip_max: float = 1.0
    seed_offset: int = 0

    def validate(self) -> None:
        if self.update_hz < 0.0:
            raise ValueError("scan.update_hz must be non-negative")
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError("scan.dropout_prob must be in [0, 1]")
        if self.stride < 1:
            raise ValueError("scan.stride must be >= 1")
        if self.clip_min >= self.clip_max:
            raise ValueError("scan clip range must be ascending")


@dataclass
class RandomizationConfig:
    enabled: bool = False
    seed: int = 42
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

    def validate(self) -> None:
        for name in (
            "ground_friction_range",
            "wheel_friction_range",
            "joint_damping_scale_range",
            "joint_friction_scale_range",
            "actuator_gain_scale_range",
            "mass_scale_range",
            "motor_response_range",
        ):
            low, high = getattr(self, name)
            if low < 0.0 or low > high:
                raise ValueError(f"randomization.{name} must be a non-negative ascending pair")
        low, high = self.action_delay_steps_range
        if low < 0 or low > high:
            raise ValueError("randomization.action_delay_steps_range must be a non-negative ascending pair")
        for name in (
            "com_offset_range",
            "imu_noise_std",
            "joint_pos_noise_std",
            "joint_vel_noise_std",
            "scan_noise_std",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"randomization.{name} must be non-negative")
        if not 0.0 <= self.scan_dropout_prob <= 1.0:
            raise ValueError("randomization.scan_dropout_prob must be in [0, 1]")


@dataclass
class SimulationConfig:
    physics_dt: float = 0.002
    control_decimation: int = 10
    duration: float = 20.0
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    impratio: float = 100.0
    solver: str = "newton"
    iterations: int = 100
    tolerance: float = 1.0e-8
    ls_iterations: int = 50
    ls_tolerance: float = 0.01
    ccd_iterations: int = 50
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    scan: ScanConfig = field(default_factory=ScanConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    fall_height: float = 0.10
    # Training terminates on bad_orientation at 1.0 rad; success still requires
    # a much stricter 0.70 tilt at the exit hold.  Do not stop at 0.85: the
    # policy is allowed to tilt and recover during the approach.
    fall_tilt: float = 1.0

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.control_decimation

    def steps(self, duration: float | None = None) -> int:
        seconds = self.duration if duration is None else duration
        if seconds <= 0.0:
            raise ValueError("duration must be positive for a finite evaluation")
        return int(round(seconds / self.control_dt))

    def validate(self) -> None:
        if self.physics_dt <= 0.0:
            raise ValueError("simulation.physics_dt must be positive")
        if self.control_decimation < 1:
            raise ValueError("simulation.control_decimation must be >= 1")
        if self.duration < 0.0:
            raise ValueError("simulation.duration must be non-negative (0 = unlimited viewer)")
        if self.integrator not in {"euler", "implicitfast"}:
            raise ValueError("simulation.integrator must be 'euler' or 'implicitfast'")
        if self.cone not in {"pyramidal", "elliptic"}:
            raise ValueError("simulation.cone must be 'pyramidal' or 'elliptic'")
        if self.solver not in {"newton", "cg", "pgs"}:
            raise ValueError("simulation.solver must be 'newton', 'cg' or 'pgs'")
        if self.fall_height < 0.0 or self.fall_tilt <= 0.0:
            raise ValueError("fall thresholds must be positive")
        self.scan.validate()
        self.randomization.validate()


@dataclass
class PolicyConfig:
    checkpoint: Path | None = None
    device: str = "cpu"
    architecture: str = "auto"
    map_shape: tuple[int, int] = (31, 13)
    use_return_mask: bool | None = None
    residual_scale: float = 0.40
    action_smoothing: float = 0.45
    action_rate_limit: float = 0.35
    action_absolute_limit: float = 4.0
    router_temperature: float = 1.0
    history_length: int | None = None
    scan_dim: int | None = None
    scan_size: tuple[float, float] | None = None
    scan_resolution: float | None = None
    strict: bool = True

    def validate(self) -> None:
        if self.architecture not in {"auto", "mlp", "ame", "moe"}:
            raise ValueError("policy.architecture must be auto|mlp|ame|moe")
        if len(self.map_shape) != 2 or min(self.map_shape) < 1:
            raise ValueError("policy.map_shape must be a positive pair")
        if self.router_temperature <= 0.0:
            raise ValueError("policy.router_temperature must be positive")
        if self.action_absolute_limit <= 0.0:
            raise ValueError("policy.action_absolute_limit must be positive")
        if not 0.0 <= self.action_smoothing <= 1.0:
            raise ValueError("policy.action_smoothing must be in [0, 1]")
        if self.action_rate_limit < 0.0:
            raise ValueError("policy.action_rate_limit must be non-negative")


@dataclass
class ObstacleConfig:
    type: str = "flat"
    difficulty: float = 0.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("obstacle.difficulty must be in [0, 1]")
        if len(self.origin) != 3:
            raise ValueError("obstacle.origin must contain x y z")


@dataclass
class EvaluationConfig:
    kind: str = "auto"
    success_distance: float | None = None
    stable_steps: int = 5
    max_tilt: float = 0.70
    min_height: float = 0.14
    min_forward_velocity: float = 0.02
    max_vertical_velocity: float = 0.75
    command_threshold: float = 0.05
    stop_on_fall: bool = True
    require_support: bool = True
    max_lateral_deviation: float | None = None
    duration: float | None = None
    plum_require_all_stones: bool = False
    plum_block_hold_steps: int = 2
    stair_reach_tolerance: float = 0.10
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.stable_steps < 1:
            raise ValueError("evaluation.stable_steps must be >= 1")
        if not 0.0 < self.max_tilt < 2.0:
            raise ValueError("evaluation.max_tilt must be in (0, 2)")
        if self.min_height < 0.0:
            raise ValueError("evaluation.min_height must be non-negative")
        if self.plum_block_hold_steps < 1:
            raise ValueError("evaluation.plum_block_hold_steps must be >= 1")


@dataclass
class OutputConfig:
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    scene_xml: Path = field(default_factory=lambda: PROJECT_ROOT / "generated" / "scene.xml")
    save_trace: bool = False
    save_failed_trace: bool = True
    trace_every: int = 10


@dataclass
class RunConfig:
    robot: RobotProfile
    robot_xml: Path
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    command: tuple[float, float, float] = (0.25, 0.0, 0.0)
    max_command: tuple[float, float, float] = (1.0, 1.0, 1.0)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    spawn_override: tuple[float, float, float, float] | None = None
    source_config: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_dim(self) -> int:
        return self.robot.frame_dim

    @property
    def history_length(self) -> int:
        return self.policy.history_length or self.robot.history_length

    @property
    def scan_dim(self) -> int:
        return self.policy.scan_dim or self.robot.scan_dim

    @property
    def scan_size(self) -> tuple[float, float]:
        if self.policy.scan_size is not None:
            return self.policy.scan_size
        return self.robot.scan_size_by_dim.get(self.scan_dim, self.robot.scan_size)

    @property
    def scan_resolution(self) -> float:
        return self.policy.scan_resolution or self.robot.scan_resolution

    def validate(self) -> None:
        self.policy.validate()
        self.simulation.validate()
        self.obstacle.validate()
        self.evaluation.validate()
        if len(self.command) != 3:
            raise ValueError("command must contain vx vy yaw")
        if len(self.max_command) != 3:
            raise ValueError("max_command must contain vx vy yaw")
        if len(self.robot.default_angles) != len(self.robot.joint_names):
            raise ValueError(
                f"{self.robot.name}: default_angles length does not match joint_names"
            )
        if len(self.robot.observation_term_names) != len(self.robot.observation_term_dims):
            raise ValueError("observation term names and dims must have the same length")
        if any(int(v) <= 0 for v in self.robot.observation_term_dims):
            raise ValueError("every observation_term_dims entry must be positive")
        if self.robot.frame_dim <= 0:
            raise ValueError("the observation frame must have a positive dimension")
        # Validate the scan grid exactly as the runtime will build it.
        width, depth = self.scan_size
        res = self.scan_resolution
        # Matches GridPatternCfg: torch.arange(-size/2, size/2 + res/2, res),
        # i.e. floor(size/res) intervals plus the centre point.
        nx = int(width / res + 1.0e-6) + 1
        ny = int(depth / res + 1.0e-6) + 1
        if nx * ny != self.scan_dim:
            raise ValueError(
                f"Scan grid {nx}x{ny}={nx * ny} does not match checkpoint scan_dim={self.scan_dim}. "
                "Set policy.scan_size or policy.scan_dim explicitly."
            )
        if self.policy.checkpoint is not None and not self.policy.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.policy.checkpoint}")
        if not self.robot_xml.exists():
            raise FileNotFoundError(f"Robot XML not found: {self.robot_xml}")

    def as_serializable(self) -> dict[str, Any]:
        """A compact, JSON-serialisable view used in result files."""

        return {
            "robot": self.robot.name,
            "robot_xml": str(self.robot_xml),
            "checkpoint": None if self.policy.checkpoint is None else str(self.policy.checkpoint),
            "obstacle": {
                "type": self.obstacle.type,
                "difficulty": self.obstacle.difficulty,
                "origin": list(self.obstacle.origin),
                "yaw": self.obstacle.yaw,
                "params": copy.deepcopy(self.obstacle.params),
            },
            "command": list(self.command),
            "duration": self.simulation.duration,
            "seed": self.simulation.randomization.seed,
            "evaluation": {
                "kind": self.evaluation.kind,
                "success_distance": self.evaluation.success_distance,
                "stable_steps": self.evaluation.stable_steps,
            },
        }


@dataclass
class SuiteCase:
    obstacle: ObstacleConfig
    command: tuple[float, float, float] | None = None
    duration: float | None = None
    evaluation: EvaluationConfig | None = None
    repeats: int | None = None
    label: str | None = None


@dataclass
class SuiteConfig:
    name: str
    cases: list[SuiteCase] = field(default_factory=list)
    repeats: int = 1
    duration: float | None = None
    command: tuple[float, float, float] | None = None
    randomization: RandomizationConfig | None = None
    seed: int = 42
    source_config: Path | None = None

    def validate(self) -> None:
        if not self.cases:
            raise ValueError("suite must contain at least one case")
        if self.repeats < 1:
            raise ValueError("suite.repeats must be >= 1")
        for case in self.cases:
            case.obstacle.validate()
            if case.evaluation is not None:
                case.evaluation.validate()


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _tuple3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _tuple2(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    return float(value[0]), float(value[1])


def _pair_float(raw: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = raw.get(key, default)
    return _tuple2(value, key)


def _pair_int(raw: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
    value = raw.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must contain two integers")
    return int(value[0]), int(value[1])


def _load_scan(raw: dict[str, Any]) -> ScanConfig:
    return ScanConfig(
        enabled=bool(raw.get("enabled", True)),
        update_hz=float(raw.get("update_hz", 10.0)),
        noise_std=max(0.0, float(raw.get("noise_std", 0.03))),
        dropout_prob=min(max(float(raw.get("dropout_prob", 0.04)), 0.0), 1.0),
        stride=max(1, int(raw.get("stride", 1))),
        max_distance=float(raw.get("max_distance", 3.0)),
        clip_min=float(raw.get("clip_min", -1.0)),
        clip_max=float(raw.get("clip_max", 1.0)),
        seed_offset=int(raw.get("seed_offset", 0)),
    )


def _load_randomization(raw: dict[str, Any] | None) -> RandomizationConfig:
    raw = raw or {}
    cfg = RandomizationConfig(
        enabled=bool(raw.get("enabled", False)),
        seed=int(raw.get("seed", 42)),
        ground_friction_range=_pair_float(raw, "ground_friction_range", (0.75, 1.25)),
        wheel_friction_range=_pair_float(raw, "wheel_friction_range", (0.70, 1.30)),
        joint_damping_scale_range=_pair_float(raw, "joint_damping_scale_range", (0.85, 1.15)),
        joint_friction_scale_range=_pair_float(raw, "joint_friction_scale_range", (0.80, 1.20)),
        actuator_gain_scale_range=_pair_float(raw, "actuator_gain_scale_range", (0.90, 1.10)),
        mass_scale_range=_pair_float(raw, "mass_scale_range", (0.95, 1.05)),
        com_offset_range=max(0.0, float(raw.get("com_offset_range", 0.015))),
        motor_response_range=_pair_float(raw, "motor_response_range", (0.75, 1.0)),
        action_delay_steps_range=_pair_int(raw, "action_delay_steps_range", (0, 2)),
        control_decimation_jitter=max(0, int(raw.get("control_decimation_jitter", 1))),
        imu_noise_std=max(0.0, float(raw.get("imu_noise_std", 0.015))),
        joint_pos_noise_std=max(0.0, float(raw.get("joint_pos_noise_std", 0.003))),
        joint_vel_noise_std=max(0.0, float(raw.get("joint_vel_noise_std", 0.03))),
        scan_noise_std=max(0.0, float(raw.get("scan_noise_std", 0.015))),
        scan_dropout_prob=min(max(float(raw.get("scan_dropout_prob", 0.02)), 0.0), 1.0),
    )
    cfg.validate()
    return cfg


def _load_simulation(raw: dict[str, Any]) -> SimulationConfig:
    scan_raw = raw.get("scan", {}) or {}
    cfg = SimulationConfig(
        physics_dt=float(raw.get("physics_dt", 0.002)),
        control_decimation=int(raw.get("control_decimation", 10)),
        duration=float(raw.get("duration", 20.0)),
        integrator=str(raw.get("integrator", "implicitfast")),
        cone=str(raw.get("cone", "elliptic")),
        impratio=float(raw.get("impratio", 100.0)),
        solver=str(raw.get("solver", "newton")),
        iterations=int(raw.get("iterations", 100)),
        tolerance=float(raw.get("tolerance", 1.0e-8)),
        ls_iterations=int(raw.get("ls_iterations", 50)),
        ls_tolerance=float(raw.get("ls_tolerance", 0.01)),
        ccd_iterations=int(raw.get("ccd_iterations", 50)),
        gravity=_tuple3(raw.get("gravity", (0.0, 0.0, -9.81)), "simulation.gravity"),
        scan=_load_scan(scan_raw),
        randomization=_load_randomization(raw.get("randomization")),
        fall_height=float(raw.get("fall_height", 0.10)),
        fall_tilt=float(raw.get("fall_tilt", 1.0)),
    )
    cfg.validate()
    return cfg


def _load_policy(raw: dict[str, Any], base: Path, profile: RobotProfile) -> PolicyConfig:
    cfg = PolicyConfig(
        checkpoint=discover_checkpoint(raw.get("checkpoint"), base),
        device=str(raw.get("device", "cpu")),
        architecture=str(raw.get("architecture", "auto")),
        map_shape=tuple(int(v) for v in raw.get("map_shape", (31, 13))),  # type: ignore[arg-type]
        use_return_mask=None if raw.get("use_return_mask") is None else bool(raw["use_return_mask"]),
        residual_scale=float(raw.get("residual_scale", 0.40)),
        action_smoothing=float(raw.get("action_smoothing", 0.45)),
        action_rate_limit=float(raw.get("action_rate_limit", 0.35)),
        action_absolute_limit=float(raw.get("action_absolute_limit", 4.0)),
        router_temperature=float(raw.get("router_temperature", 1.0)),
        history_length=None if raw.get("history_length") is None else int(raw["history_length"]),
        scan_dim=None if raw.get("scan_dim") is None else int(raw["scan_dim"]),
        scan_size=None if raw.get("scan_size") is None else _tuple2(raw["scan_size"], "policy.scan_size"),
        scan_resolution=None if raw.get("scan_resolution") is None else float(raw["scan_resolution"]),
        strict=bool(raw.get("strict", True)),
    )
    if cfg.history_length is None:
        cfg.history_length = profile.history_length
    cfg.validate()
    return cfg


def _load_evaluation(raw: dict[str, Any] | None) -> EvaluationConfig:
    raw = raw or {}
    cfg = EvaluationConfig(
        kind=str(raw.get("kind", "auto")),
        success_distance=None if raw.get("success_distance") is None else float(raw["success_distance"]),
        stable_steps=int(raw.get("stable_steps", 5)),
        max_tilt=float(raw.get("max_tilt", 0.70)),
        min_height=float(raw.get("min_height", 0.14)),
        min_forward_velocity=float(raw.get("min_forward_velocity", 0.02)),
        max_vertical_velocity=float(raw.get("max_vertical_velocity", 0.75)),
        command_threshold=float(raw.get("command_threshold", 0.05)),
        stop_on_fall=bool(raw.get("stop_on_fall", True)),
        require_support=bool(raw.get("require_support", True)),
        max_lateral_deviation=(
            None if raw.get("max_lateral_deviation") is None else float(raw["max_lateral_deviation"])
        ),
        duration=None if raw.get("duration") is None else float(raw["duration"]),
        plum_require_all_stones=bool(raw.get("plum_require_all_stones", False)),
        plum_block_hold_steps=int(raw.get("plum_block_hold_steps", 2)),
        stair_reach_tolerance=float(raw.get("stair_reach_tolerance", 0.10)),
        extra=dict(raw.get("extra", {})),
    )
    cfg.validate()
    return cfg


def _load_obstacle(raw: dict[str, Any] | None) -> ObstacleConfig:
    raw = raw or {}
    cfg = ObstacleConfig(
        type=str(raw.get("type", "flat")),
        difficulty=float(raw.get("difficulty", 0.0)),
        origin=_tuple3(raw.get("origin", (0.0, 0.0, 0.0)), "obstacle.origin"),
        yaw=float(raw.get("yaw", 0.0)),
        params=dict(raw.get("params", {})),
    )
    cfg.validate()
    return cfg


def _legacy_to_modern(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate the old flat config.yaml schema into the new nested schema."""

    joint_count = len(raw.get("joint_names", ()))
    profile_name = "minidog" if joint_count and joint_count <= 12 else "go2w"
    spawn = raw.get("spawn", (0.0, 0.0, 0.45, 0.0))
    lane_y = float(spawn[1]) if len(spawn) >= 2 else 0.0
    if abs(lane_y - 3.35) < 0.25:
        obstacle_type = "plum"
    elif abs(lane_y - 0.85) < 0.25:
        obstacle_type = "high_wall"
    elif abs(lane_y + 0.85) < 0.25:
        obstacle_type = "low_bar"
    elif abs(lane_y + 3.35) < 0.25:
        obstacle_type = "gap"
    else:
        obstacle_type = "flat"
    randomization = dict(raw.get("randomization", {}) or {})
    scan = {
        "enabled": True,
        "update_hz": 10.0 / max(1, int(raw.get("scan_update_interval", 5))),
        "noise_std": float(randomization.get("scan_noise_std", 0.03)),
        "dropout_prob": float(randomization.get("scan_dropout_prob", 0.04)),
        "stride": int(raw.get("scan_stride", 1)),
    }
    return {
        "robot": {"profile": profile_name, "xml": raw.get("robot_xml")},
        "policy": {
            "checkpoint": raw.get("checkpoint"),
            "scan_dim": raw.get("scan_dim"),
            "scan_size": raw.get("scan_size"),
        },
        "simulation": {
            "physics_dt": raw.get("physics_dt", 0.002),
            "control_decimation": raw.get("control_decimation", 10),
            "duration": raw.get("duration", 20.0),
            "scan": scan,
            "randomization": randomization,
        },
        "obstacle": {"type": obstacle_type, "difficulty": 0.35},
        "command": raw.get("command", (0.25, 0.0, 0.0)),
        "max_command": raw.get("max_command", (1.0, 1.0, 1.0)),
        "evaluation": {"kind": "auto"},
        "spawn_override": spawn,
    }


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "robot_xml" in raw:
        raw = _legacy_to_modern(raw)

    base = path.parent
    robot_raw = dict(raw.get("robot", {}) or {})
    profile = get_profile(str(robot_raw.get("profile", "go2w")))
    robot_xml = discover_robot_xml(profile, robot_raw.get("xml"), base)

    policy = _load_policy(dict(raw.get("policy", {}) or {}), base, profile)
    simulation = _load_simulation(dict(raw.get("simulation", {}) or {}))
    obstacle = _load_obstacle(dict(raw.get("obstacle", {}) or {}))
    command = _tuple3(raw.get("command", (0.25, 0.0, 0.0)), "command")
    max_command = _tuple3(raw.get("max_command", (1.0, 1.0, 1.0)), "max_command")
    evaluation = _load_evaluation(dict(raw.get("evaluation", {}) or {}))
    output_raw = dict(raw.get("output", {}) or {})
    output = OutputConfig(
        results_dir=resolve_path(output_raw.get("results_dir", "results"), base) or (PROJECT_ROOT / "results"),
        scene_xml=resolve_path(output_raw.get("scene_xml", "generated/scene.xml"), base)
        or (PROJECT_ROOT / "generated" / "scene.xml"),
        save_trace=bool(output_raw.get("save_trace", False)),
        save_failed_trace=bool(output_raw.get("save_failed_trace", True)),
        trace_every=max(1, int(output_raw.get("trace_every", 10))),
    )
    spawn_override = raw.get("spawn_override")
    spawn_tuple = None
    if spawn_override is not None:
        if not isinstance(spawn_override, (list, tuple)) or len(spawn_override) != 4:
            raise ValueError("spawn_override must contain x y z yaw")
        spawn_tuple = tuple(float(v) for v in spawn_override)  # type: ignore[assignment]

    cfg = RunConfig(
        robot=profile,
        robot_xml=robot_xml,
        policy=policy,
        simulation=simulation,
        obstacle=obstacle,
        command=command,
        max_command=max_command,
        evaluation=evaluation,
        output=output,
        spawn_override=spawn_tuple,
        source_config=path,
        raw=raw,
    )
    cfg.validate()
    return cfg


def load_suite(path: str | Path) -> SuiteConfig:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Suite config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base = path.parent
    cases: list[SuiteCase] = []
    for index, item in enumerate(raw.get("cases", ())):
        if not isinstance(item, dict):
            raise ValueError(f"suite.cases[{index}] must be a mapping")
        obstacle_raw = dict(item.get("obstacle", item))
        obstacle = _load_obstacle(obstacle_raw)
        command = None
        if item.get("command") is not None:
            command = _tuple3(item["command"], f"cases[{index}].command")
        evaluation = None
        if item.get("evaluation") is not None:
            evaluation = _load_evaluation(dict(item["evaluation"]))
        repeats = None if item.get("repeats") is None else int(item["repeats"])
        cases.append(
            SuiteCase(
                obstacle=obstacle,
                command=command,
                duration=None if item.get("duration") is None else float(item["duration"]),
                evaluation=evaluation,
                repeats=repeats,
                label=item.get("label"),
            )
        )
    randomization = None
    if raw.get("randomization") is not None:
        randomization = _load_randomization(dict(raw["randomization"]))
    suite = SuiteConfig(
        name=str(raw.get("name", path.stem)),
        cases=cases,
        repeats=int(raw.get("repeats", 1)),
        duration=None if raw.get("duration") is None else float(raw["duration"]),
        command=None if raw.get("command") is None else _tuple3(raw["command"], "suite.command"),
        randomization=randomization,
        seed=int(raw.get("seed", 42)),
        source_config=path,
    )
    suite.validate()
    return suite


def dump_config(config: RunConfig) -> dict[str, Any]:
    return config.as_serializable()
