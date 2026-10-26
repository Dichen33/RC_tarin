"""Quaternion helpers shared by observation and evaluation code.

MuJoCo stores quaternions as ``(w, x, y, z)``.  The training code uses the
same convention through ``mjlab``/``rsl_rl``, so all helpers here keep it.
"""

from __future__ import annotations

import math

import numpy as np

Vec3 = np.ndarray
Quat = np.ndarray


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-9:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    return q / norm


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame."""

    q = normalize_quat(q)
    w, x, y, z = q
    qv = np.asarray((x, y, z), dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a body-frame vector into the world frame."""

    q = normalize_quat(q)
    w, x, y, z = q
    qv = np.asarray((x, y, z), dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def yaw_from_quat(q: np.ndarray) -> float:
    q = normalize_quat(q)
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def projected_gravity(q: np.ndarray) -> np.ndarray:
    """Gravity direction in the body frame (training observation term)."""

    return quat_rotate_inverse(q, np.asarray((0.0, 0.0, -1.0), dtype=np.float64))


def body_linear_velocity(q: np.ndarray, world_velocity: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(q, np.asarray(world_velocity, dtype=np.float64))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)), dtype=np.float64)


def local_to_world(
    origin: tuple[float, float, float],
    yaw: float,
    x: float,
    y: float,
    z: float = 0.0,
) -> tuple[float, float, float]:
    """Rotate a lane-local offset by yaw and translate it to world coordinates."""

    c, s = math.cos(yaw), math.sin(yaw)
    return (
        float(origin[0] + c * x - s * y),
        float(origin[1] + s * x + c * y),
        float(origin[2] + z),
    )
