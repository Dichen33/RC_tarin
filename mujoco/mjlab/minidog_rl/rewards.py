from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


RewardFn = Callable[[object, np.ndarray, object], float]


@dataclass(frozen=True)
class RewardTerm:
    name: str
    func: RewardFn
    weight: float


def track_lin_vel(env: object, action: np.ndarray, base: object) -> float:
    del action
    vel_err = base.vel_body[:2] - env.command[:2]
    return float(np.exp(-6.0 * np.dot(vel_err, vel_err)))


def track_yaw(env: object, action: np.ndarray, base: object) -> float:
    del action
    yaw_err = base.omega_body[2] - env.command[2]
    return float(np.exp(-2.0 * yaw_err * yaw_err))


def base_height(env: object, action: np.ndarray, base: object) -> float:
    del action, base
    target = env.cfg.height
    return float(np.exp(-80.0 * (env.data.qpos[2] - target) ** 2))


def orientation(env: object, action: np.ndarray, base: object) -> float:
    del env, action
    roll, pitch, _ = base.rpy
    return float(np.exp(-5.0 * (roll * roll + pitch * pitch)))


def action_rate(env: object, action: np.ndarray, base: object) -> float:
    del base
    return -float(np.mean((action - env.last_action) ** 2))


def joint_torques(env: object, action: np.ndarray, base: object) -> float:
    del action, base
    return -float(np.mean(np.square(env.data.ctrl)))


def vertical_velocity(env: object, action: np.ndarray, base: object) -> float:
    del env, action
    return -float(base.vel_body[2] * base.vel_body[2])


def roll_pitch_penalty(env: object, action: np.ndarray, base: object) -> float:
    del env, action
    roll, pitch, _ = base.rpy
    return -float(roll * roll + pitch * pitch)


def crawl_height(env: object, action: np.ndarray, base: object) -> float:
    del action, base
    return float(np.exp(-120.0 * (env.data.qpos[2] - env.cfg.height) ** 2))


def terrain_level_bonus(env: object, action: np.ndarray, base: object) -> float:
    del action, base
    if not getattr(env.cfg, "terrain_curriculum", False):
        return 0.0
    denom = max(1, int(env.cfg.terrain_levels) - 1)
    return float(env.current_terrain_level / denom)


def make_reward_terms(profile: str) -> tuple[RewardTerm, ...]:
    profile = profile.lower().strip()
    common = (
        RewardTerm("reward_track_lin_vel", track_lin_vel, 1.4),
        RewardTerm("reward_track_yaw", track_yaw, 0.5),
        RewardTerm("reward_height", base_height, 0.4),
        RewardTerm("reward_orientation", orientation, 0.4),
        RewardTerm("reward_action_rate", action_rate, 0.03),
        RewardTerm("reward_joint_torques", joint_torques, 0.0002),
    )
    if profile == "rough":
        return (
            RewardTerm("reward_track_lin_vel", track_lin_vel, 1.8),
            RewardTerm("reward_track_yaw", track_yaw, 0.6),
            RewardTerm("reward_height", base_height, 0.25),
            RewardTerm("reward_orientation", orientation, 0.45),
            RewardTerm("reward_vertical_velocity", vertical_velocity, 0.15),
            RewardTerm("reward_action_rate", action_rate, 0.02),
            RewardTerm("reward_joint_torques", joint_torques, 0.00015),
            RewardTerm("reward_terrain_level", terrain_level_bonus, 0.15),
        )
    if profile == "crawl":
        return (
            RewardTerm("reward_track_lin_vel", track_lin_vel, 1.2),
            RewardTerm("reward_track_yaw", track_yaw, 0.4),
            RewardTerm("reward_height", crawl_height, 0.8),
            RewardTerm("reward_orientation", orientation, 0.25),
            RewardTerm("reward_roll_pitch_penalty", roll_pitch_penalty, 0.10),
            RewardTerm("reward_action_rate", action_rate, 0.02),
            RewardTerm("reward_joint_torques", joint_torques, 0.00015),
            RewardTerm("reward_terrain_level", terrain_level_bonus, 0.10),
        )
    return common


def compute_reward(env: object, action: np.ndarray, base: object, terms: tuple[RewardTerm, ...]) -> tuple[float, dict[str, float]]:
    values: dict[str, float] = {}
    reward = 0.0
    for term in terms:
        value = term.func(env, action, base)
        values[term.name] = value
        reward += term.weight * value
    if env._fallen():
        reward -= 5.0
    return float(reward), values

