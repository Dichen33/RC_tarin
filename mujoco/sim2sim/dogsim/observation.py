"""Observation pipeline: proprioception terms, history and height scan.

The important compatibility details live here:

- height = vertical ray distance from the base frame, not a signed world Z;
- misses are the clipped maximum range (1.0 after clipping), not zero;
- the scan is held at 10 Hz by default, matching the ODIN1-style training
  observation, while physics/control runs at 50 Hz;
- the actor receives term-major history, exactly like ``actor_proprio``;
- the return-validity channel is reconstructed in the policy from values
  below 0.999, so the scan itself stays on the training contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Deque
from collections import deque

import mujoco
import numpy as np

from .config import RunConfig
from .mathutils import projected_gravity, quat_rotate_inverse, yaw_from_quat
from .robots import RobotProfile
from .scene import Scene


@dataclass
class ScanResult:
    values: np.ndarray
    refreshed: bool
    valid_fraction: float


class HeightScanner:
    def __init__(self, scene: Scene, cfg: RunConfig) -> None:
        self.scene = scene
        self.cfg = cfg
        self.robot = cfg.robot
        self.model = scene.model
        self.data = scene.data
        self.width, self.depth = cfg.scan_size
        self.resolution = cfg.scan_resolution
        self.scan_dim = cfg.scan_dim
        self.xs = np.arange(-self.width / 2.0, self.width / 2.0 + 1.0e-6, self.resolution, dtype=np.float32)
        self.ys = np.arange(-self.depth / 2.0, self.depth / 2.0 + 1.0e-6, self.resolution, dtype=np.float32)
        if len(self.xs) * len(self.ys) != self.scan_dim:
            raise ValueError(
                f"Scan grid {len(self.xs)}x{len(self.ys)} does not match scan_dim={self.scan_dim}"
            )
        self.nx, self.ny = len(self.xs), len(self.ys)
        self.points = np.asarray([(x, y) for x in self.xs for y in self.ys], dtype=np.float32)
        self.scan_cfg = cfg.simulation.scan
        self.control_dt = cfg.simulation.control_dt
        self.update_interval = max(
            1, int(round(1.0 / (self.scan_cfg.update_hz * self.control_dt)))
        ) if self.scan_cfg.update_hz > 0.0 else 1
        self.geom_mask = np.zeros(6, dtype=np.uint8)
        for group in self.robot.scan_include_geom_groups:
            self.geom_mask[int(group)] = 1
        if not np.any(self.geom_mask):
            raise ValueError("scan_include_geom_groups must contain at least one group")
        self._cache: np.ndarray | None = None
        self._age = 0
        self._step = 0
        self._rng = np.random.default_rng(cfg.simulation.randomization.seed + self.scan_cfg.seed_offset)
        self._history: Deque[np.ndarray] = deque(maxlen=cfg.history_length)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed + self.scan_cfg.seed_offset)
        self._cache = None
        self._age = 0
        self._step = 0
        self._history.clear()

    def _sample_points(self) -> tuple[list[int], list[int]]:
        stride = max(1, int(self.scan_cfg.stride))
        sample_x = list(range(0, self.nx, stride))
        sample_y = list(range(0, self.ny, stride))
        if sample_x[-1] != self.nx - 1:
            sample_x.append(self.nx - 1)
        if sample_y[-1] != self.ny - 1:
            sample_y.append(self.ny - 1)
        return sample_x, sample_y

    def _cast(self) -> np.ndarray:
        base = np.asarray(self.data.qpos[:3], dtype=np.float64)
        yaw = yaw_from_quat(self.data.qpos[3:7])
        cy, sy = math.cos(yaw), math.sin(yaw)
        sample_x, sample_y = self._sample_points()
        sampled = np.ones((self.nx, self.ny), dtype=np.float32)
        for ix in sample_x:
            for iy in sample_y:
                x, y = float(self.xs[ix]), float(self.ys[iy])
                origin = base + np.asarray((cy * x - sy * y, sy * x + cy * y, 0.0), dtype=np.float64)
                geomid = np.asarray([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
                    self.geom_mask,
                    True,
                    self.scene.base_body_id,
                    geomid,
                )
                if distance >= 0.0:
                    # The ray starts at the base frame's Z.  Training reports
                    # frame_z - hit_z, which for a vertical ray is the
                    # distance itself.  A miss keeps the max-range sentinel.
                    sampled[ix, iy] = float(distance)
                else:
                    sampled[ix, iy] = float(self.scan_cfg.max_distance)

        if sample_x == list(range(0, self.nx)) and sample_y == list(range(0, self.ny)):
            values = sampled.reshape(-1)
        else:
            values = np.ones((self.nx, self.ny), dtype=np.float32)
            for ix in range(self.nx):
                nearest_x = min(sample_x, key=lambda value: abs(value - ix))
                for iy in range(self.ny):
                    nearest_y = min(sample_y, key=lambda value: abs(value - iy))
                    values[ix, iy] = sampled[nearest_x, nearest_y]
            values = values.reshape(-1)

        values = np.clip(values, self.scan_cfg.clip_min, self.scan_cfg.clip_max)
        # Scan noise/dropout are part of the actor observation contract in
        # training (``safe_height_scan`` always applies them to the actor,
        # while the critic sees the clean scan).  A randomize run may widen
        # them; it must not be the only source.
        noise = float(self.scan_cfg.noise_std)
        dropout = float(self.scan_cfg.dropout_prob)
        if self.cfg.simulation.randomization.enabled:
            noise = max(noise, float(self.cfg.simulation.randomization.scan_noise_std))
            dropout = max(dropout, float(self.cfg.simulation.randomization.scan_dropout_prob))
        if noise > 0.0:
            values = values + self._rng.normal(0.0, noise, size=values.shape).astype(np.float32)
        if dropout > 0.0:
            dropped = self._rng.random(values.shape) < dropout
            values[dropped] = 1.0
        values = np.clip(values, self.scan_cfg.clip_min, self.scan_cfg.clip_max).astype(np.float32)
        return values

    def update(self) -> ScanResult:
        if self._cache is None or not self.scan_cfg.enabled:
            refresh = True
        else:
            # Count policy steps since the last scan.  With a 10 Hz scan and a
            # 50 Hz policy this refreshes on calls 1, 6, 11, ... which matches
            # ``episode_length_buf % 5 == 0`` in training.
            self._age += 1
            refresh = self._age >= self.update_interval
        if refresh:
            if self.scan_cfg.enabled:
                self._cache = self._cast()
            else:
                self._cache = np.ones(self.scan_dim, dtype=np.float32)
            self._age = 0
        assert self._cache is not None
        values = self._cache.copy()
        self._history.append(values)
        valid = float(np.mean(values < 0.999))
        self._step += 1
        return ScanResult(values=values, refreshed=refresh, valid_fraction=valid)

    def scan_history(self, history_length: int | None = None) -> np.ndarray:
        """Return a fixed-size scan history, repeating the newest frame."""

        length = history_length or self.cfg.history_length
        if not self._history:
            raise RuntimeError("HeightScanner.update() must be called before scan_history()")
        frames = list(self._history)
        while len(frames) < length:
            frames.insert(0, frames[0])
        frames = frames[-length:]
        return np.stack(frames, axis=0)


class ObservationPipeline:
    def __init__(self, scene: Scene, cfg: RunConfig) -> None:
        self.scene = scene
        self.cfg = cfg
        self.robot = cfg.robot
        self.control_dt = cfg.simulation.control_dt
        self.scale_ang_vel = 0.25
        self.scale_joint_vel = 0.05
        self.default_angles = np.asarray(self.robot.default_angles, dtype=np.float32)
        self.history: Deque[np.ndarray] = deque(maxlen=cfg.history_length)
        self._rng = np.random.default_rng(cfg.simulation.randomization.seed + 101)
        self._elapsed = 0.0
        self._command = np.asarray(cfg.command, dtype=np.float32)
        self._executed_action = np.zeros(self.robot.action_dim, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed + 101)
        self.history.clear()
        self._elapsed = 0.0
        self._executed_action = np.zeros(self.robot.action_dim, dtype=np.float32)

    def set_command(self, command: np.ndarray) -> None:
        self._command = np.asarray(command, dtype=np.float32)

    def set_executed_action(self, action: np.ndarray) -> None:
        self._executed_action = np.asarray(action, dtype=np.float32)

    def advance(self, dt: float | None = None) -> None:
        self._elapsed += self.control_dt if dt is None else float(dt)

    def _gait_phase(self, command: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(command[:2]) + abs(command[2]))
        active = norm > self.robot.gait_command_threshold
        freq = min(
            max(self.robot.gait_base_frequency + norm, self.robot.gait_base_frequency),
            self.robot.gait_max_frequency,
        )
        frequency = freq if active else 0.0
        phase = (self._elapsed * frequency) % 1.0
        angle = 2.0 * math.pi * phase
        return np.asarray((math.sin(angle), math.cos(angle)), dtype=np.float32)

    def terms(self) -> dict[str, np.ndarray]:
        data = self.scene.data
        q = data.qpos[3:7]
        ang_vel = quat_rotate_inverse(q, data.qvel[3:6]).astype(np.float32) * self.scale_ang_vel
        gravity = projected_gravity(q).astype(np.float32)
        command = self._command.astype(np.float32)
        gait = self._gait_phase(command)
        qpos = data.qpos[self.scene.joints.qpos_ids].astype(np.float32)
        qvel = data.qvel[self.scene.joints.qvel_ids].astype(np.float32)
        joint_pos = (qpos - self.default_angles).astype(np.float32)
        joint_vel = (qvel * self.scale_joint_vel).astype(np.float32)
        last_action = self._executed_action.astype(np.float32)
        rand = self.cfg.simulation.randomization
        if rand.enabled:
            ang_vel = ang_vel + self._rng.normal(0.0, rand.imu_noise_std, ang_vel.shape).astype(np.float32)
            gravity = gravity + self._rng.normal(0.0, rand.imu_noise_std, gravity.shape).astype(np.float32)
            joint_pos = joint_pos + self._rng.normal(0.0, rand.joint_pos_noise_std, joint_pos.shape).astype(np.float32)
            joint_vel = joint_vel + self._rng.normal(0.0, rand.joint_vel_noise_std, joint_vel.shape).astype(np.float32)
        return {
            "base_ang_vel": ang_vel,
            "projected_gravity": gravity,
            "command": command,
            "gait_phase": gait,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "last_action": last_action,
        }

    def frame(self) -> np.ndarray:
        terms = self.terms()
        pieces: list[np.ndarray] = []
        for name, expected in zip(self.robot.observation_term_names, self.robot.observation_term_dims):
            value = terms[name]
            if value.size != expected:
                raise ValueError(
                    f"Observation term {name} has {value.size} values, expected {expected}"
                )
            pieces.append(value.reshape(-1))
        frame = np.concatenate(pieces).astype(np.float32)
        if frame.size != self.robot.frame_dim:
            raise AssertionError(f"Frame has {frame.size} values, expected {self.robot.frame_dim}")
        return frame

    def update(self) -> np.ndarray:
        frame = self.frame()
        if not self.history:
            # Match training reset behaviour: the entire history starts at the
            # first observable frame instead of zeros.
            for _ in range(self.cfg.history_length):
                self.history.append(frame.copy())
        else:
            self.history.append(frame.copy())
        return frame

    def term_major_history(self) -> np.ndarray:
        if not self.history:
            raise RuntimeError("ObservationPipeline.update() must be called first")
        stacked = np.stack(list(self.history), axis=0)  # [H, frame_dim]
        pieces: list[np.ndarray] = []
        offset = 0
        for dim in self.robot.observation_term_dims:
            term = stacked[:, offset : offset + dim]
            pieces.append(term.reshape(-1))
            offset += dim
        result = np.concatenate(pieces).astype(np.float32)
        expected = self.robot.frame_dim * self.cfg.history_length
        if result.size != expected:
            raise AssertionError(f"History has {result.size} values, expected {expected}")
        return result
