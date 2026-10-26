from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from minidog_rl.paths import MODEL_PATH, ROBOT_ROOT
from minidog_rl.rewards import compute_reward, make_reward_terms
from minidog_rl.terrain import TerrainSpec, resolve_scene_path


if str(ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_ROOT))

from ik_control import (  # noqa: E402
    JOINT_ORDER,
    LEGS,
    STAND_BASE_HEIGHT,
    MotionCommand,
    VMCPositionIKController,
    apply_joint_pd,
    read_base_state,
)


@dataclass
class CommandRange:
    vx: tuple[float, float] = (-0.12, 0.16)
    vy: tuple[float, float] = (-0.09, 0.09)
    yaw_rate: tuple[float, float] = (-0.8, 0.8)


@dataclass
class MiniDogEnvConfig:
    task_id: str = "Robot-Flat-v0"
    episode_seconds: float = 8.0
    control_dt: float = 0.02
    ik_iterations: int = 8
    action_scale: np.ndarray = field(
        default_factory=lambda: np.tile(np.array([0.20, 0.28, 0.28], dtype=np.float32), 4)
    )
    command_range: CommandRange = field(default_factory=CommandRange)
    fixed_command: tuple[float, float, float] | None = None
    height: float = STAND_BASE_HEIGHT
    reward_profile: str = "flat"
    terrain_mode: str = "plane"
    terrain_profile: str = "mixed"
    terrain_level: int = 0
    terrain_curriculum: bool = False
    terrain_levels: int = 5
    terrain_types: tuple[str, ...] = ("mixed",)
    terrain_seed: int = 0
    terrain_nrow: int = 128
    terrain_ncol: int = 128
    terrain_width_m: float = 4.0
    terrain_length_m: float = 4.0
    scene_path: str | None = None
    terminate_height: float = 0.13
    terminate_rp: float = 0.95
    seed: int | None = None


class MiniDogResidualEnv(gym.Env):
    """Gymnasium environment for residual learning over the IK gait controller.

    Action: 12 normalized residual joint target offsets in the order
    FL/FR/RL/RR x hip/thigh/calf.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config: MiniDogEnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or MiniDogEnvConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(self.cfg.seed)
        self.viewer = None
        self.current_terrain_level = int(np.clip(self.cfg.terrain_level, 0, max(0, self.cfg.terrain_levels - 1)))
        self.current_terrain_type = self.cfg.terrain_profile
        self.reward_terms = make_reward_terms(self.cfg.reward_profile)

        self.scene_path = self._resolve_scene_path()
        self._load_scene(self.scene_path)
        self.action_scale = np.asarray(self.cfg.action_scale, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        obs_dim = 51
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.sim_time = 0.0
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        self.episode_start_xy = np.zeros(2, dtype=np.float32)

    def _resolve_scene_path(self) -> Path:
        if self.cfg.scene_path:
            return Path(self.cfg.scene_path)
        profile = self.current_terrain_type if self.cfg.terrain_curriculum else self.cfg.terrain_profile
        level = self.current_terrain_level if self.cfg.terrain_curriculum else self.cfg.terrain_level
        return resolve_scene_path(
            TerrainSpec(
                mode=self.cfg.terrain_mode,
                profile=profile,
                level=level,
                seed=self.cfg.terrain_seed if self.cfg.terrain_seed is not None else (self.cfg.seed or 0),
                nrow=self.cfg.terrain_nrow,
                ncol=self.cfg.terrain_ncol,
                width_m=self.cfg.terrain_width_m,
                length_m=self.cfg.terrain_length_m,
            )
        )

    def _load_scene(self, scene_path: Path) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.controller = VMCPositionIKController(self.model, ik_iterations=self.cfg.ik_iterations)
        self.dt = float(self.model.opt.timestep)
        self.ctrl_steps = max(1, int(round(self.cfg.control_dt / self.dt)))
        self.max_steps = int(round(self.cfg.episode_seconds / self.cfg.control_dt))
        self.joint_ids = [
            self.controller.ik.joint_ids[leg][joint_index]
            for leg in LEGS
            for joint_index, _ in enumerate(JOINT_ORDER)
        ]
        self.qpos_adrs = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=int)
        self.dof_adrs = np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=int)

    def _update_terrain_curriculum(self) -> None:
        if not self.cfg.terrain_curriculum or self.step_count <= 0:
            return
        distance = float(np.linalg.norm(self.data.qpos[:2] - self.episode_start_xy))
        expected = float(np.linalg.norm(self.command[:2]) * self.cfg.episode_seconds)
        terrain_len = max(0.5, float(self.cfg.terrain_length_m))
        move_up = distance > 0.45 * terrain_len and distance > 0.35 * expected
        move_down = expected > 0.2 and distance < max(0.2, 0.25 * expected) and not move_up
        if move_up:
            self.current_terrain_level += 1
        elif move_down:
            self.current_terrain_level -= 1
        self.current_terrain_level = int(np.clip(self.current_terrain_level, 0, max(0, self.cfg.terrain_levels - 1)))
        terrain_types = tuple(self.cfg.terrain_types) or (self.cfg.terrain_profile,)
        self.current_terrain_type = str(self.rng.choice(terrain_types))

    def _reload_scene_if_needed(self) -> None:
        next_scene = self._resolve_scene_path()
        if next_scene != self.scene_path:
            self._load_scene(next_scene)

    def _sample_command(self) -> np.ndarray:
        if self.cfg.fixed_command is not None:
            return np.asarray(self.cfg.fixed_command, dtype=np.float32)
        return np.array(
            [
                self.rng.uniform(*self.cfg.command_range.vx),
                self.rng.uniform(*self.cfg.command_range.vy),
                self.rng.uniform(*self.cfg.command_range.yaw_rate),
            ],
            dtype=np.float32,
        )

    def _motion_command(self) -> MotionCommand:
        vx, vy, yaw_rate = self.command
        mode = "stand" if np.linalg.norm(self.command) < 1e-3 else "trot"
        return MotionCommand(mode=mode, vx=float(vx), vy=float(vy), yaw_rate=float(yaw_rate), height=self.cfg.height)

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[self.qpos_adrs].copy(), self.data.qvel[self.dof_adrs].copy()

    def _get_obs(self) -> np.ndarray:
        base = read_base_state(self.data)
        q, qd = self._joint_state()
        gravity_body = base.rot.T @ np.array([0.0, 0.0, -1.0], dtype=float)
        obs = np.concatenate(
            [
                gravity_body,
                base.rpy,
                base.vel_body,
                base.omega_body,
                self.command,
                q,
                qd,
                self.last_action,
            ]
        )
        return obs.astype(np.float32)

    def _residual_q_des(self, action: np.ndarray) -> dict[int, float]:
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        q_des = dict(self.controller.q_des)
        residual = self.action_scale * action
        for joint_id, dq in zip(self.joint_ids, residual):
            lo, hi = self.model.jnt_range[joint_id]
            q_des[joint_id] = float(np.clip(q_des[joint_id] + dq, lo, hi))
        return q_des

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        base = read_base_state(self.data)
        return compute_reward(self, action, base, self.reward_terms)

    def _fallen(self) -> bool:
        base = read_base_state(self.data)
        return bool(
            self.data.qpos[2] < self.cfg.terminate_height
            or abs(base.rpy[0]) > self.cfg.terminate_rp
            or abs(base.rpy[1]) > self.cfg.terminate_rp
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._update_terrain_curriculum()
        self._reload_scene_if_needed()
        self.controller.reset(self.data)
        self.step_count = 0
        self.sim_time = 0.0
        self.command = self._sample_command()
        if options:
            if "command" in options:
                self.command = np.asarray(options["command"], dtype=np.float32)
        self.last_action[:] = 0.0
        self.last_ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.episode_start_xy = self.data.qpos[:2].copy()
        return self._get_obs(), self._info(0.0, {})

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        for _ in range(self.ctrl_steps):
            self.controller.step(self.data, self._motion_command(), self.sim_time)
            residual_q_des = self._residual_q_des(action)
            apply_joint_pd(self.model, self.data, residual_q_des)
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt
        reward, reward_terms = self._reward(action)
        self.step_count += 1
        terminated = self._fallen()
        truncated = self.step_count >= self.max_steps
        info = self._info(reward, reward_terms)
        self.last_action = action.copy()
        self.last_ctrl = self.data.ctrl.copy()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def _info(self, reward: float, reward_terms: dict[str, float]) -> dict[str, Any]:
        base = read_base_state(self.data)
        vel_err = base.vel_body[:2] - self.command[:2]
        yaw_err = base.omega_body[2] - self.command[2]
        bad_orientation = abs(base.rpy[0]) > self.cfg.terminate_rp or abs(base.rpy[1]) > self.cfg.terminate_rp
        base_ground_contact = self.data.qpos[2] < self.cfg.terminate_height
        time_out = self.step_count >= self.max_steps
        return {
            "reward": reward,
            "command": self.command.copy(),
            "base_xy": self.data.qpos[:2].copy(),
            "base_z": float(self.data.qpos[2]),
            "yaw": float(base.rpy[2]),
            "body_vel": base.vel_body.copy(),
            "body_omega": base.omega_body.copy(),
            "fallen": self._fallen(),
            "error_vel_xy": float(np.linalg.norm(vel_err)),
            "error_vel_yaw": float(abs(yaw_err)),
            "termination_time_out": float(time_out),
            "termination_bad_orientation": float(bad_orientation),
            "termination_base_ground_contact": float(base_ground_contact),
            "termination_nan_detection": 0.0,
            "curriculum_max": float(self.current_terrain_level),
            "terrain_level": float(self.current_terrain_level),
            "terrain_type": self.current_terrain_type,
            **reward_terms,
        }

    def render(self):
        if self.viewer is None:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
