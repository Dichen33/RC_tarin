"""Per-step metrics shared by all obstacle evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..config import RunConfig
from ..mathutils import body_linear_velocity, projected_gravity, yaw_from_quat
from ..scene import Scene


@dataclass
class StepState:
    step: int
    time: float
    base_pos: np.ndarray
    base_quat: np.ndarray
    base_velocity: np.ndarray
    base_z_rel: float
    tilt: float
    forward_velocity: float
    vertical_velocity: float
    support_contact: bool
    illegal_contact: bool
    base_contact: bool
    thigh_contact: bool
    calf_contact: bool
    foot_contacts: np.ndarray
    foot_positions: np.ndarray
    foot_velocities: np.ndarray
    collision_pairs: int
    fallen: bool
    along: float

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "time": round(self.time, 4),
            "base_pos": [round(float(v), 5) for v in self.base_pos],
            "base_z_rel": round(float(self.base_z_rel), 5),
            "tilt": round(float(self.tilt), 5),
            "forward_velocity": round(float(self.forward_velocity), 5),
            "vertical_velocity": round(float(self.vertical_velocity), 5),
            "support": bool(self.support_contact),
            "illegal": bool(self.illegal_contact),
            "base_contact": bool(self.base_contact),
            "thigh_contact": bool(self.thigh_contact),
            "calf_contact": bool(self.calf_contact),
            "foot_contacts": [bool(v) for v in self.foot_contacts],
            "fallen": bool(self.fallen),
            "along": round(float(self.along), 5),
        }


class MetricsTracker:
    def __init__(self, scene: Scene, cfg: RunConfig) -> None:
        self.scene = scene
        self.cfg = cfg
        self.model = scene.model
        self.data = scene.data
        self.robot = cfg.robot
        self.terrain_origin_z = float(scene.build.terrain_origin[2])
        self.spawn = np.asarray(scene.build.spawn[:3], dtype=np.float64)
        self.spawn_yaw = float(scene.build.spawn[3])
        self.foot_body_ids = [int(v) for v in scene.foot_body_ids]
        self.foot_body_set = set(self.foot_body_ids)
        self.base_body_id = scene.base_body_id
        self.thigh_body_ids = self._body_ids(self.robot.thigh_bodies)
        self.calf_body_ids = self._body_ids(self.robot.calf_bodies)
        self.illegal_body_ids = {self.base_body_id, *self.thigh_body_ids}
        self._pair_set: set[tuple[int, int]] = set()
        self.min_height = float("inf")
        self.max_tilt = 0.0
        self.max_forward_velocity = 0.0
        self.stationary_steps = 0
        self._previous_pos: np.ndarray | None = None
        self.peak_height = 0.0

    def _body_ids(self, names: tuple[str, ...]) -> set[int]:
        ids: set[int] = set()
        for name in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                ids.add(int(bid))
        return ids

    def reset(self) -> None:
        self._pair_set.clear()
        self.min_height = float("inf")
        self.max_tilt = 0.0
        self.max_forward_velocity = 0.0
        self.stationary_steps = 0
        self._previous_pos = None
        self.peak_height = 0.0

    def update(self, step: int, time: float) -> StepState:
        data = self.data
        qpos = np.asarray(data.qpos[:7], dtype=np.float64)
        qvel = np.asarray(data.qvel[:6], dtype=np.float64)
        base_pos = qpos[:3].copy()
        base_quat = qpos[3:7].copy()
        gravity = projected_gravity(base_quat)
        tilt = float(np.linalg.norm(gravity[:2]))
        base_vel = body_linear_velocity(base_quat, qvel[:3])
        base_z_rel = float(base_pos[2] - self.terrain_origin_z)
        vertical_vel = float(abs(qvel[2]))
        # Progress is measured along the lane, not along the robot's current
        # heading: a yawed robot must not shrink its own displacement.
        along = float(
            (base_pos[0] - self.spawn[0]) * np.cos(self.spawn_yaw)
            + (base_pos[1] - self.spawn[1]) * np.sin(self.spawn_yaw)
        )
        foot_positions = np.asarray(
            [data.xpos[bid].copy() for bid in self.foot_body_ids], dtype=np.float64
        )
        foot_velocities = np.asarray(
            [np.asarray(data.cvel[bid, 3:6], dtype=np.float64) for bid in self.foot_body_ids],
            dtype=np.float64,
        )
        foot_contacts = np.zeros(len(self.foot_body_ids), dtype=bool)
        support = False
        illegal = False
        base_contact = False
        thigh_contact = False
        calf_contact = False
        pair_count = 0
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if body1 in self.foot_body_set or body2 in self.foot_body_set:
                other_geom = geom2 if body1 in self.foot_body_set else geom1
                if other_geom in self.scene.terrain_geom_ids:
                    if body1 in self.foot_body_set:
                        # Foot order in scene.foot_body_ids is policy order.
                        for foot_index, foot_bid in enumerate(self.foot_body_ids):
                            if body1 == foot_bid:
                                foot_contacts[foot_index] = True
                    if body2 in self.foot_body_set:
                        for foot_index, foot_bid in enumerate(self.foot_body_ids):
                            if body2 == foot_bid:
                                foot_contacts[foot_index] = True
                    support = True
            robot_body = None
            other_geom = None
            if body1 in self.scene.robot_body_ids and body2 not in self.scene.robot_body_ids:
                robot_body, other_geom = body1, geom2
            elif body2 in self.scene.robot_body_ids and body1 not in self.scene.robot_body_ids:
                robot_body, other_geom = body2, geom1
            if robot_body is not None and other_geom in self.scene.terrain_geom_ids:
                pair = (min(geom1, geom2), max(geom1, geom2))
                if pair not in self._pair_set:
                    self._pair_set.add(pair)
                    pair_count += 1
                if robot_body == self.base_body_id:
                    base_contact = True
                    illegal = True
                elif robot_body in self.thigh_body_ids:
                    thigh_contact = True
                    illegal = True
                elif robot_body in self.calf_body_ids:
                    # Shank/calf contact is deliberately allowed by the
                    # training success criterion; it is still reported.
                    calf_contact = True

        fallen = bool(
            base_z_rel < self.cfg.simulation.fall_height
            or tilt > self.cfg.simulation.fall_tilt
            or base_contact
        )
        self.min_height = min(self.min_height, base_z_rel)
        self.max_tilt = max(self.max_tilt, tilt)
        self.max_forward_velocity = max(self.max_forward_velocity, float(base_vel[0]))
        self.peak_height = max(self.peak_height, base_z_rel)
        if abs(float(base_vel[0])) < 0.01:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0
        state = StepState(
            step=step,
            time=time,
            base_pos=base_pos,
            base_quat=base_quat,
            base_velocity=base_vel,
            base_z_rel=base_z_rel,
            tilt=tilt,
            forward_velocity=float(base_vel[0]),
            vertical_velocity=vertical_vel,
            support_contact=support,
            illegal_contact=illegal,
            base_contact=base_contact,
            thigh_contact=thigh_contact,
            calf_contact=calf_contact,
            foot_contacts=foot_contacts,
            foot_positions=foot_positions,
            foot_velocities=foot_velocities,
            collision_pairs=pair_count,
            fallen=fallen,
            along=along,
        )
        self._previous_pos = base_pos
        return state
