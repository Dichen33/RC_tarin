import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "scene.xml"

LEGS = ("FL", "FR", "RL", "RR")
JOINT_ORDER = ("hip", "thigh", "calf")
FOOT_RADIUS = 0.01675
STRAIGHT_LEG_BASE_HEIGHT = 0.262
STAND_BASE_HEIGHT = 0.235
DOWN_BASE_HEIGHT = 0.165
STAND_THIGH_SEED = 0.48
STAND_CALF_SEED = -0.95

NOMINAL_FOOT_XY = {
    "FL": np.array([0.11415, 0.1142]),
    "FR": np.array([0.11415, -0.1142]),
    "RL": np.array([-0.11415, 0.1142]),
    "RR": np.array([-0.11415, -0.1142]),
}
NOMINAL_FOOT_BASE = {}

KP = {
    "hip": 18.0,
    "thigh": 32.0,
    "calf": 32.0,
}
KD = {
    "hip": 0.55,
    "thigh": 0.95,
    "calf": 0.95,
}


@dataclass
class MotionCommand:
    mode: str = "stand"
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    height: float = STAND_BASE_HEIGHT


@dataclass
class LegPhase:
    phase: float
    swing_phase: float
    stance_phase: float
    in_swing: bool


@dataclass
class BaseState:
    pos: np.ndarray
    rot: np.ndarray
    rpy: np.ndarray
    vel_body: np.ndarray
    omega_body: np.ndarray


def nominal_foot_targets(base_height: float) -> dict[str, np.ndarray]:
    return {
        leg: np.array([xy[0], xy[1], -(base_height - FOOT_RADIUS)], dtype=float)
        for leg, xy in NOMINAL_FOOT_XY.items()
    }


NOMINAL_FOOT_BASE = nominal_foot_targets(STAND_BASE_HEIGHT)
DOWN_FOOT_BASE = nominal_foot_targets(DOWN_BASE_HEIGHT)


def smoothstep(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def read_base_state(data: mujoco.MjData) -> BaseState:
    rot = quat_to_matrix(data.qpos[3:7])
    return BaseState(
        pos=data.qpos[:3].copy(),
        rot=rot,
        rpy=matrix_to_rpy(rot),
        vel_body=rot.T @ data.qvel[:3],
        omega_body=rot.T @ data.qvel[3:6],
    )


class GaitPlanner:
    def __init__(self, period: float = 0.86, duty_factor: float = 0.72):
        self.period = period
        self.duty_factor = duty_factor
        self.trot_offsets = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}
        self.lateral_offsets = self.trot_offsets

    def _phase_offsets(self, command: MotionCommand) -> dict[str, float]:
        if command.mode != "trot":
            return self.trot_offsets
        if abs(command.vy) > max(0.035, 1.15 * abs(command.vx)):
            return self.lateral_offsets
        return self.trot_offsets

    def leg_phase(self, leg: str, t: float, command: MotionCommand) -> LegPhase:
        mode = command.mode
        if mode != "trot":
            return LegPhase(phase=0.0, swing_phase=0.0, stance_phase=0.0, in_swing=False)

        phase = (t / self.period + self._phase_offsets(command)[leg]) % 1.0
        in_swing = phase >= self.duty_factor
        if in_swing:
            swing_phase = (phase - self.duty_factor) / (1.0 - self.duty_factor)
            stance_phase = 1.0
        else:
            swing_phase = 0.0
            stance_phase = phase / self.duty_factor
        return LegPhase(phase=phase, swing_phase=swing_phase, stance_phase=stance_phase, in_swing=in_swing)


class FootTrajectoryGenerator:
    def __init__(self, planner: GaitPlanner):
        self.planner = planner
        self.max_step_xy = np.array([0.095, 0.034], dtype=float)
        self.lateral_max_step_xy = np.array([0.060, 0.040], dtype=float)
        self.lateral_min_step_y = 0.026
        self.forward_velocity_gain = 1.75
        self.lateral_velocity_gain = 1.70
        self.lateral_forward_bias_gain = 2.00
        self.lateral_forward_bias_low_gain = 0.75
        self.stance_anchor_blend = 0.0
        self.clearance = 0.022
        self.max_height_rate = 0.35
        self.filtered_height = STAND_BASE_HEIGHT
        self.last_t: Optional[float] = None
        self.desired_yaw: Optional[float] = None
        self.raibert_gain = np.array([0.16, 0.22], dtype=float)
        self.yaw_kp = 3.2
        self.yaw_kd = 0.75
        self.roll_gain = 0.018
        self.roll_rate_gain = 0.006
        self.pitch_gain = 0.020
        self.pitch_rate_gain = 0.008
        self.velocity_tilt_gain = np.array([0.010, 0.012], dtype=float)
        self.stance_anchor_world: dict[str, Optional[np.ndarray]] = {leg: None for leg in LEGS}
        self.prev_in_swing = {leg: False for leg in LEGS}

    def _clear_stance_anchors(self) -> None:
        for leg in LEGS:
            self.stance_anchor_world[leg] = None
            self.prev_in_swing[leg] = False

    def _filtered_targets(self, command: MotionCommand, t: float) -> dict[str, np.ndarray]:
        target_height = DOWN_BASE_HEIGHT if command.mode == "down" else command.height
        target_height = float(np.clip(target_height, DOWN_BASE_HEIGHT, STRAIGHT_LEG_BASE_HEIGHT))

        if self.last_t is None:
            self.filtered_height = target_height
            self.last_t = t
        else:
            dt = max(0.0, min(0.05, t - self.last_t))
            max_step = self.max_height_rate * dt
            self.filtered_height += float(np.clip(target_height - self.filtered_height, -max_step, max_step))
            self.last_t = t

        return nominal_foot_targets(self.filtered_height)

    def _foot_body_velocity(self, leg: str, command: MotionCommand) -> np.ndarray:
        xy = NOMINAL_FOOT_XY[leg]
        yaw_component = command.yaw_rate * np.array([-xy[1], xy[0]], dtype=float)
        lateral_alpha = float(np.clip((abs(command.vy) - 0.06) / 0.04, 0.0, 1.0))
        lateral_bias_gain = (
            self.lateral_forward_bias_low_gain
            + lateral_alpha * (self.lateral_forward_bias_gain - self.lateral_forward_bias_low_gain)
        )
        lateral_forward_bias = (
            lateral_bias_gain * abs(command.vy)
            if abs(command.vy) > max(0.035, 1.15 * abs(command.vx))
            else 0.0
        )
        return np.array(
            [
                self.forward_velocity_gain * command.vx + lateral_forward_bias,
                self.lateral_velocity_gain * command.vy,
            ],
            dtype=float,
        ) + yaw_component

    def _is_stationary_trot(self, command: MotionCommand) -> bool:
        return (
            command.mode == "trot"
            and abs(command.vx) < 0.015
            and abs(command.vy) < 0.015
            and abs(command.yaw_rate) < 0.04
        )

    def _closed_loop_command(self, command: MotionCommand, t: float, base_state: Optional[BaseState]) -> MotionCommand:
        if base_state is None:
            return command

        if self.desired_yaw is None or command.mode == "down":
            self.desired_yaw = float(base_state.rpy[2])

        dt = 0.0 if self.last_t is None else max(0.0, min(0.05, t - self.last_t))
        if abs(command.yaw_rate) > 1e-3:
            self.desired_yaw = wrap_to_pi(self.desired_yaw + command.yaw_rate * dt)

        yaw_error = wrap_to_pi(self.desired_yaw - float(base_state.rpy[2]))
        lateral_weight = min(1.0, abs(command.vy) / 0.09)
        yaw_kp = self.yaw_kp + 1.3 * lateral_weight
        yaw_kd = self.yaw_kd + 0.55 * lateral_weight
        yaw_hold = yaw_kp * yaw_error - yaw_kd * float(base_state.omega_body[2])
        yaw_rate = command.yaw_rate + (0.0 if abs(command.yaw_rate) > 1e-3 else yaw_hold)
        yaw_rate = float(np.clip(yaw_rate, -2.6, 2.6))

        desired_vel = np.array([command.vx, command.vy], dtype=float)
        vel_error = desired_vel - base_state.vel_body[:2]
        corrected_vel = desired_vel + self.raibert_gain * vel_error
        if abs(command.vx) < 0.015:
            corrected_vel[0] = 0.0
        else:
            corrected_vel[0] = np.clip(corrected_vel[0], -abs(command.vx), abs(command.vx))
        if abs(command.vy) < 0.015:
            corrected_vel[1] = 0.0
        else:
            corrected_vel[1] = np.clip(corrected_vel[1], -abs(command.vy), abs(command.vy))
        corrected_vel = np.clip(corrected_vel, [-0.20, -0.10], [0.20, 0.10])

        return MotionCommand(
            mode=command.mode,
            vx=float(corrected_vel[0]),
            vy=float(corrected_vel[1]),
            yaw_rate=yaw_rate,
            height=command.height,
        )

    def _attitude_compensation(self, leg: str, command: MotionCommand, base_state: Optional[BaseState]) -> np.ndarray:
        if base_state is None or command.mode == "down":
            return np.zeros(3)

        xy = NOMINAL_FOOT_XY[leg]
        roll, pitch, _ = base_state.rpy
        roll_rate, pitch_rate = base_state.omega_body[0], base_state.omega_body[1]
        comp = np.zeros(3)
        comp[0] += -self.pitch_gain * pitch - self.pitch_rate_gain * pitch_rate
        comp[1] += self.roll_gain * roll + self.roll_rate_gain * roll_rate
        comp[2] += -0.10 * pitch * xy[0] + 0.10 * roll * xy[1]

        desired_vel = np.array([command.vx, command.vy], dtype=float)
        vel_error = base_state.vel_body[:2] - desired_vel
        comp[:2] += self.velocity_tilt_gain * vel_error
        return np.clip(comp, [-0.025, -0.025, -0.012], [0.025, 0.025, 0.012])

    def targets(self, command: MotionCommand, t: float, base_state: Optional[BaseState] = None) -> dict[str, np.ndarray]:
        neutral = self._filtered_targets(command, t)
        if command.mode in ("stand", "down") or self._is_stationary_trot(command):
            self._clear_stance_anchors()
            return {
                leg: neutral[leg] + self._attitude_compensation(leg, command, base_state)
                for leg in LEGS
            }

        command = self._closed_loop_command(command, t, base_state)
        targets = {}
        for leg in LEGS:
            leg_phase = self.planner.leg_phase(leg, t, command)
            foot_velocity = self._foot_body_velocity(leg, command)
            max_step_xy = self.max_step_xy.copy()
            if abs(command.vy) > max(0.035, 1.15 * abs(command.vx)):
                max_step_xy = self.lateral_max_step_xy.copy()
                lateral_alpha = float(np.clip((abs(command.vy) - 0.06) / 0.04, 0.0, 1.0))
                max_step_xy[1] = self.lateral_min_step_y + lateral_alpha * (
                    self.lateral_max_step_xy[1] - self.lateral_min_step_y
                )
            step_xy = np.clip(foot_velocity * self.planner.period, -max_step_xy, max_step_xy)
            target = neutral[leg].copy()
            stance_widen = min(0.035, 0.26 * abs(command.vy))
            target[1] += np.sign(NOMINAL_FOOT_XY[leg][1]) * stance_widen

            if leg_phase.in_swing:
                self.stance_anchor_world[leg] = None
                s = smoothstep(leg_phase.swing_phase)
                target[:2] += step_xy * (s - 0.5)
                target[2] += self.clearance * np.sin(np.pi * leg_phase.swing_phase)
            else:
                s = leg_phase.stance_phase
                open_loop_target = target.copy()
                open_loop_target[:2] += step_xy * (0.5 - s)
                if base_state is not None:
                    if self.stance_anchor_world[leg] is None or self.prev_in_swing[leg]:
                        touchdown = target.copy()
                        touchdown[:2] += step_xy * 0.5
                        self.stance_anchor_world[leg] = base_state.pos + base_state.rot @ touchdown
                    anchor_target = base_state.rot.T @ (self.stance_anchor_world[leg] - base_state.pos)
                    anchor_target[2] = open_loop_target[2]
                    anchor_target[:2] = neutral[leg][:2] + np.clip(
                        anchor_target[:2] - neutral[leg][:2],
                        -1.25 * max_step_xy,
                        1.25 * max_step_xy,
                    )
                    target = (1.0 - self.stance_anchor_blend) * open_loop_target + self.stance_anchor_blend * anchor_target
                else:
                    target = open_loop_target
            target += self._attitude_compensation(leg, command, base_state)
            targets[leg] = target
            self.prev_in_swing[leg] = leg_phase.in_swing
        return targets


class QuadrupedIK:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_ids = {
            leg: [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint}_joint")
                for joint in JOINT_ORDER
            ]
            for leg in LEGS
        }
        self.site_ids = {
            leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
            for leg in LEGS
        }

        for leg in LEGS:
            if any(joint_id < 0 for joint_id in self.joint_ids[leg]):
                raise RuntimeError(f"Missing joint in leg {leg}")
            if self.site_ids[leg] < 0:
                raise RuntimeError(f"Missing foot site in leg {leg}")

        self.qpos_adrs = {
            joint_id: int(model.jnt_qposadr[joint_id])
            for leg in LEGS
            for joint_id in self.joint_ids[leg]
        }
        self.dof_adrs = {
            joint_id: int(model.jnt_dofadr[joint_id])
            for leg in LEGS
            for joint_id in self.joint_ids[leg]
        }
        self.ranges = {
            joint_id: np.array(model.jnt_range[joint_id])
            for leg in LEGS
            for joint_id in self.joint_ids[leg]
        }

    def seed_from_model(self) -> dict[int, float]:
        q = {
            joint_id: float(self.model.qpos0[self.qpos_adrs[joint_id]])
            for leg in LEGS
            for joint_id in self.joint_ids[leg]
        }
        for leg in LEGS:
            thigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_thigh_joint")
            calf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_calf_joint")
            q[thigh_id] = STAND_THIGH_SEED
            q[calf_id] = STAND_CALF_SEED
        return q

    def _set_scratch_pose(self, q: dict[int, float]) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:7] = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
        self.data.qvel[:] = 0
        for joint_id, value in q.items():
            self.data.qpos[self.qpos_adrs[joint_id]] = value
        mujoco.mj_forward(self.model, self.data)

    def solve(
        self,
        foot_targets_base: dict[str, np.ndarray],
        seed: Optional[dict[int, float]] = None,
        iterations: int = 40,
        damping: float = 1e-4,
    ) -> dict[int, float]:
        q = self.seed_from_model() if seed is None else dict(seed)

        for _ in range(iterations):
            self._set_scratch_pose(q)
            max_error = 0.0

            for leg in LEGS:
                target = foot_targets_base[leg]
                site_id = self.site_ids[leg]
                err = target - self.data.site_xpos[site_id]
                max_error = max(max_error, float(np.linalg.norm(err)))

                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, None, site_id)
                cols = [self.dof_adrs[joint_id] for joint_id in self.joint_ids[leg]]
                jac = jacp[:, cols]
                lhs = jac @ jac.T + damping * np.eye(3)
                dq = jac.T @ np.linalg.solve(lhs, err)
                dq = np.clip(dq, -0.08, 0.08)

                for joint_id, step in zip(self.joint_ids[leg], dq):
                    lo, hi = self.ranges[joint_id]
                    q[joint_id] = float(np.clip(q[joint_id] + step, lo, hi))

            if max_error < 1e-5:
                break

        return q

    def foot_positions(self, q: dict[int, float]) -> dict[str, np.ndarray]:
        self._set_scratch_pose(q)
        return {
            leg: self.data.site_xpos[self.site_ids[leg]].copy()
            for leg in LEGS
        }


def gait_targets(
    mode: str,
    t: float,
    vx: float = 0.0,
    vy: float = 0.0,
    yaw_rate: float = 0.0,
    height: float = STAND_BASE_HEIGHT,
    base_state: Optional[BaseState] = None,
) -> dict[str, np.ndarray]:
    planner = GaitPlanner()
    trajectory = FootTrajectoryGenerator(planner)
    command = MotionCommand(mode=mode, vx=vx, vy=vy, yaw_rate=yaw_rate, height=height)
    return trajectory.targets(command, t, base_state)


def joint_kind(joint_name: str) -> str:
    for kind in JOINT_ORDER:
        if kind in joint_name:
            return kind
    return "thigh"


def apply_joint_pd(model: mujoco.MjModel, data: mujoco.MjData, q_des: dict[int, float]) -> float:
    max_ctrl = 0.0
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        kind = joint_kind(joint_name)
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        torque = KP[kind] * (q_des[joint_id] - data.qpos[qadr]) - KD[kind] * data.qvel[dadr]
        lo, hi = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(torque, lo, hi)
        max_ctrl = max(max_ctrl, abs(float(data.ctrl[actuator_id])))
    return max_ctrl


def reset_to_ik_stand(model: mujoco.MjModel, data: mujoco.MjData, q_des: dict[int, float]) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[:7] = np.array([0, 0, STAND_BASE_HEIGHT, 1, 0, 0, 0], dtype=float)
    data.qvel[:] = 0
    data.ctrl[:] = 0
    for joint_id, q in q_des.items():
        data.qpos[model.jnt_qposadr[joint_id]] = q
    mujoco.mj_forward(model, data)


class VMCPositionIKController:
    def __init__(self, model: mujoco.MjModel, ik_iterations: int = 10):
        self.model = model
        self.ik = QuadrupedIK(model)
        self.planner = GaitPlanner()
        self.trajectory = FootTrajectoryGenerator(self.planner)
        self.ik_iterations = ik_iterations
        self.q_des = self.ik.solve(NOMINAL_FOOT_BASE)
        self.last_targets = {leg: pos.copy() for leg, pos in NOMINAL_FOOT_BASE.items()}
        self.max_ctrl = 0.0
        self.filtered_command = MotionCommand()
        self.last_t: Optional[float] = None
        self.max_vel_rate = np.array([0.35, 0.22], dtype=float)
        self.max_yaw_rate_rate = 1.8
        self.max_height_rate = 0.30

    def reset(self, data: mujoco.MjData) -> None:
        self.q_des = self.ik.solve(NOMINAL_FOOT_BASE, seed=self.q_des, iterations=40)
        reset_to_ik_stand(self.model, data, self.q_des)
        self.trajectory.filtered_height = STAND_BASE_HEIGHT
        self.trajectory.last_t = None
        self.trajectory.desired_yaw = None
        self.trajectory._clear_stance_anchors()
        self.filtered_command = MotionCommand()
        self.last_t = None
        self.max_ctrl = 0.0

    def zero_motion(self, data: mujoco.MjData, t: float, mode: str = "stand") -> None:
        base_state = read_base_state(data)
        self.filtered_command = MotionCommand(mode=mode, height=self.trajectory.filtered_height)
        self.trajectory.desired_yaw = float(base_state.rpy[2])
        self.trajectory.last_t = t
        self.trajectory._clear_stance_anchors()
        self.last_t = t
        data.qvel[:6] *= 0.15

    def _filter_command(self, command: MotionCommand, t: float) -> MotionCommand:
        if command.mode in ("stand", "down"):
            self.filtered_command.mode = command.mode
            self.filtered_command.vx = 0.0
            self.filtered_command.vy = 0.0
            self.filtered_command.yaw_rate = 0.0
            self.filtered_command.height = command.height
            self.last_t = t
            return MotionCommand(
                self.filtered_command.mode,
                self.filtered_command.vx,
                self.filtered_command.vy,
                self.filtered_command.yaw_rate,
                self.filtered_command.height,
            )

        dt = 0.0 if self.last_t is None else max(0.0, min(0.05, t - self.last_t))
        self.last_t = t
        self.filtered_command.mode = command.mode

        desired_vel = np.array(
            [
                np.clip(command.vx, -0.18, 0.18),
                np.clip(command.vy, -0.09, 0.09),
            ],
            dtype=float,
        )
        current_vel = np.array([self.filtered_command.vx, self.filtered_command.vy], dtype=float)
        max_step = self.max_vel_rate * dt
        current_vel += np.clip(desired_vel - current_vel, -max_step, max_step)
        self.filtered_command.vx = float(current_vel[0])
        self.filtered_command.vy = float(current_vel[1])

        desired_yaw = float(np.clip(command.yaw_rate, -1.0, 1.0))
        yaw_step = self.max_yaw_rate_rate * dt
        self.filtered_command.yaw_rate += float(np.clip(desired_yaw - self.filtered_command.yaw_rate, -yaw_step, yaw_step))

        desired_height = float(np.clip(command.height, DOWN_BASE_HEIGHT, STRAIGHT_LEG_BASE_HEIGHT))
        height_step = self.max_height_rate * dt
        self.filtered_command.height += float(np.clip(desired_height - self.filtered_command.height, -height_step, height_step))

        return MotionCommand(
            self.filtered_command.mode,
            self.filtered_command.vx,
            self.filtered_command.vy,
            self.filtered_command.yaw_rate,
            self.filtered_command.height,
        )

    def step(self, data: mujoco.MjData, command: MotionCommand, t: float) -> float:
        command = self._filter_command(command, t)
        base_state = read_base_state(data)
        self.last_targets = self.trajectory.targets(command, t, base_state)
        self.q_des = self.ik.solve(self.last_targets, seed=self.q_des, iterations=self.ik_iterations)
        ctrl = apply_joint_pd(self.model, data, self.q_des)
        self.max_ctrl = max(self.max_ctrl, ctrl)
        return ctrl


def print_ik_solution(model: mujoco.MjModel, ik: QuadrupedIK, q_des: dict[int, float]) -> None:
    print("IK stand joint targets:")
    for leg in LEGS:
        values = []
        for joint_id in ik.joint_ids[leg]:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            values.append(f"{name}={q_des[joint_id]: .5f}")
        print("  " + "  ".join(values))

    print("Foot target errors in base frame:")
    foot_positions = ik.foot_positions(q_des)
    for leg in LEGS:
        err = foot_positions[leg] - NOMINAL_FOOT_BASE[leg]
        print(f"  {leg}: {np.linalg.norm(err):.6e} m  pos={np.round(foot_positions[leg], 5)}")


def run_headless(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    controller = VMCPositionIKController(model, ik_iterations=args.ik_iterations)
    if args.period is not None:
        controller.planner.period = args.period
    if args.lateral_gain is not None:
        controller.trajectory.lateral_velocity_gain = args.lateral_gain
    if args.lateral_bias is not None:
        controller.trajectory.lateral_forward_bias_gain = args.lateral_bias
    if args.lateral_step_y is not None:
        controller.trajectory.lateral_max_step_xy[1] = args.lateral_step_y
    if args.lateral_step_x is not None:
        controller.trajectory.lateral_max_step_xy[0] = args.lateral_step_x
    controller.reset(data)

    if args.print_ik:
        print_ik_solution(model, controller.ik, controller.q_des)

    command = MotionCommand(args.mode, args.vx, args.vy, args.yaw_rate, args.height)
    steps = int(args.seconds / model.opt.timestep)
    start_xy = data.qpos[:2].copy()
    log_interval = float(args.log_interval)
    next_log_t = 0.0
    if log_interval > 0.0:
        print("time,x,y,z,yaw,body_vx,body_vy,body_wz,contacts")
    for step in range(steps):
        t = step * model.opt.timestep
        controller.step(data, command, t)
        mujoco.mj_step(model, data)
        if log_interval > 0.0 and t + model.opt.timestep >= next_log_t:
            base_state = read_base_state(data)
            print(
                f"{t:.3f},{data.qpos[0]:.4f},{data.qpos[1]:.4f},{data.qpos[2]:.4f},"
                f"{base_state.rpy[2]:.4f},{base_state.vel_body[0]:.4f},"
                f"{base_state.vel_body[1]:.4f},{base_state.omega_body[2]:.4f},{data.ncon}"
            )
            next_log_t += log_interval

    base_state = read_base_state(data)
    displacement_xy = data.qpos[:2] - start_xy
    avg_vel_xy = displacement_xy / max(args.seconds, model.opt.timestep)
    print(f"model: {MODEL_PATH}")
    print(f"mode={args.mode} seconds={args.seconds} vx={args.vx} vy={args.vy} yaw_rate={args.yaw_rate} height={args.height}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")
    print(f"final_base_xy={np.round(data.qpos[:2], 4)} m")
    print(f"displacement_xy={np.round(displacement_xy, 4)} m")
    print(f"avg_vel_xy={np.round(avg_vel_xy, 4)} m/s")
    print(f"final_base_z={data.qpos[2]:.4f} m")
    print(f"final_yaw={base_state.rpy[2]:.4f} rad")
    print(f"body_vel={np.round(base_state.vel_body, 4)} m/s")
    print(f"body_omega={np.round(base_state.omega_body, 4)} rad/s")
    print(f"max_abs_ctrl={controller.max_ctrl:.3f} Nm")
    print(f"contacts={data.ncon}")
    print(f"fallen={data.qpos[2] < 0.12}")


def run_viewer(args: argparse.Namespace) -> None:
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    controller = VMCPositionIKController(model, ik_iterations=args.ik_iterations)
    controller.reset(data)
    command = MotionCommand(args.mode, args.vx, args.vy, args.yaw_rate, args.height)

    with viewer.launch_passive(model, data) as handle:
        step = 0
        while handle.is_running():
            t = step * model.opt.timestep
            controller.step(data, command, t)
            mujoco.mj_step(model, data)
            handle.sync()
            step += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IK foot placement and PD torque control for HTDW-4438 in MuJoCo.")
    parser.add_argument("--mode", choices=("stand", "trot", "down"), default="stand")
    parser.add_argument("--vx", type=float, default=0.0, help="Commanded forward velocity in m/s for trot mode.")
    parser.add_argument("--vy", type=float, default=0.0, help="Commanded lateral velocity in m/s for trot mode.")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Commanded yaw rate in rad/s for trot mode.")
    parser.add_argument("--height", type=float, default=STAND_BASE_HEIGHT, help="Commanded standing base height in m.")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--ik-iterations", type=int, default=12)
    parser.add_argument("--log-interval", type=float, default=0.0, help="Print CSV trajectory samples every N seconds.")
    parser.add_argument("--period", type=float, default=None, help="Override gait period in seconds for tuning.")
    parser.add_argument("--lateral-gain", type=float, default=None, help="Override lateral foot velocity gain for tuning.")
    parser.add_argument("--lateral-bias", type=float, default=None, help="Override lateral forward bias gain for tuning.")
    parser.add_argument("--lateral-step-y", type=float, default=None, help="Override lateral max y step in meters for tuning.")
    parser.add_argument("--lateral-step-x", type=float, default=None, help="Override lateral max x step in meters for tuning.")
    parser.add_argument("--print-ik", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.viewer:
        run_viewer(args)
    else:
        run_headless(args)


if __name__ == "__main__":
    main()
