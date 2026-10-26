"""Reset-time domain randomization, kept independent of terrain and policy."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import RunConfig
from .scene import Scene


@dataclass
class ControlParams:
    motor_response: float = 1.0
    action_delay_steps: int = 0
    control_steps: int = 10


class Randomizer:
    def __init__(self, scene: Scene, cfg: RunConfig) -> None:
        self.scene = scene
        self.cfg = cfg
        self.model = scene.model
        self.randomization = cfg.simulation.randomization
        self._base_geom_friction = self.model.geom_friction.copy()
        self._base_dof_damping = self.model.dof_damping.copy()
        self._base_dof_frictionloss = self.model.dof_frictionloss.copy()
        self._base_actuator_gainprm = self.model.actuator_gainprm.copy()
        self._base_actuator_biasprm = self.model.actuator_biasprm.copy()
        self._base_body_mass = self.model.body_mass.copy()
        self._base_body_ipos = self.model.body_ipos.copy()
        self._rng = np.random.default_rng(self.randomization.seed)

    def _uniform(self, values: tuple[float, float]) -> float:
        return float(self._rng.uniform(values[0], values[1]))

    def _restore(self) -> None:
        self.model.geom_friction[:] = self._base_geom_friction
        self.model.dof_damping[:] = self._base_dof_damping
        self.model.dof_frictionloss[:] = self._base_dof_frictionloss
        self.model.actuator_gainprm[:] = self._base_actuator_gainprm
        self.model.actuator_biasprm[:] = self._base_actuator_biasprm
        self.model.body_mass[:] = self._base_body_mass
        self.model.body_ipos[:] = self._base_body_ipos

    def reset(self, seed: int | None = None) -> ControlParams:
        self._rng = np.random.default_rng(self.randomization.seed if seed is None else seed)
        self._restore()
        cfg = self.randomization
        if not cfg.enabled:
            mujoco.mj_setConst(self.model, self.scene.data)
            return ControlParams(
                motor_response=1.0,
                action_delay_steps=0,
                control_steps=self.cfg.simulation.control_decimation,
            )

        ground_scale = self._uniform(cfg.ground_friction_range)
        for geom_id in self.scene.terrain_geom_ids:
            self.model.geom_friction[geom_id, 0] *= ground_scale
        wheel_scale = self._uniform(cfg.wheel_friction_range)
        for geom_id in self.scene.foot_geom_ids:
            self.model.geom_friction[geom_id, 0] *= wheel_scale

        damping_scale = self._uniform(cfg.joint_damping_scale_range)
        friction_scale = self._uniform(cfg.joint_friction_scale_range)
        self.model.dof_damping[self.scene.joints.qvel_ids] *= damping_scale
        self.model.dof_frictionloss[self.scene.joints.qvel_ids] *= friction_scale

        gain_scale = self._uniform(cfg.actuator_gain_scale_range)
        actuator_ids = np.concatenate(
            (self.scene.joints.position_actuator_ids, self.scene.joints.velocity_actuator_ids)
        ).astype(np.int32)
        self.model.actuator_gainprm[actuator_ids, 0] *= gain_scale
        # Keep the position/velocity servo bias consistent with its gain.
        # Scaling gainprm alone would leave the feedback zero point shifted.
        self.model.actuator_biasprm[actuator_ids, 1] *= gain_scale
        self.model.actuator_biasprm[actuator_ids, 2] *= gain_scale

        mass_scale = self._uniform(cfg.mass_scale_range)
        robot_body_ids = np.asarray(sorted(self.scene.robot_body_ids), dtype=np.int32)
        self.model.body_mass[robot_body_ids] *= mass_scale
        if cfg.com_offset_range > 0.0:
            offset = self._rng.uniform(-cfg.com_offset_range, cfg.com_offset_range, size=2)
            self.model.body_ipos[self.scene.base_body_id, :2] += offset
        mujoco.mj_setConst(self.model, self.scene.data)

        low_delay, high_delay = cfg.action_delay_steps_range
        delay = int(self._rng.integers(low_delay, high_delay + 1))
        jitter = cfg.control_decimation_jitter
        control_steps = self.cfg.simulation.control_decimation
        if jitter:
            control_steps += int(self._rng.integers(-jitter, jitter + 1))
        control_steps = max(1, control_steps)
        return ControlParams(
            motor_response=self._uniform(cfg.motor_response_range),
            action_delay_steps=delay,
            control_steps=control_steps,
        )
