from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minidog_rl.paths import MODEL_PATH, ROBOT_ROOT


TERRAIN_VERSION = "v1"


@dataclass(frozen=True)
class TerrainSpec:
    mode: str = "flat"
    profile: str = "mixed"
    level: int = 0
    seed: int = 0
    nrow: int = 128
    ncol: int = 128
    width_m: float = 4.0
    length_m: float = 4.0


def resolve_scene_path(spec: TerrainSpec) -> Path:
    mode = spec.mode.lower().strip()
    if mode in {"flat", "plane", "none"}:
        return MODEL_PATH
    if mode not in {"rough", "terrain", "curriculum"}:
        raise ValueError(f"Unknown terrain mode: {spec.mode}")

    payload = {
        "version": TERRAIN_VERSION,
        "mode": mode,
        "profile": spec.profile.lower().strip(),
        "level": int(spec.level),
        "seed": int(spec.seed),
        "nrow": int(spec.nrow),
        "ncol": int(spec.ncol),
        "width_m": float(spec.width_m),
        "length_m": float(spec.length_m),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    terrain_dir = ROBOT_ROOT
    terrain_dir.mkdir(parents=True, exist_ok=True)

    scene_path = terrain_dir / f"terrain_{digest}.xml"
    hfield_path = terrain_dir / "meshes" / f"terrain_{digest}.hfield"
    hfield_path.parent.mkdir(parents=True, exist_ok=True)
    if scene_path.exists() and hfield_path.exists():
        return scene_path

    surface = _build_surface(payload["profile"], payload["level"], payload["seed"], payload["nrow"], payload["ncol"])
    _write_hfield(hfield_path, surface)
    _write_scene(scene_path, hfield_path.name, spec, surface)
    return scene_path


def _build_surface(profile: str, level: int, seed: int, nrow: int, ncol: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = max(0, int(level))
    x = np.linspace(-1.0, 1.0, nrow, dtype=np.float32)[:, None]
    y = np.linspace(-1.0, 1.0, ncol, dtype=np.float32)[None, :]

    height_span = 0.05 + 0.02 * level
    surface = np.zeros((nrow, ncol), dtype=np.float32)

    if profile == "slope_up":
        surface = np.broadcast_to(0.9 * height_span * x, (nrow, ncol)).copy()
    elif profile == "slope_down":
        surface = np.broadcast_to(-0.9 * height_span * x, (nrow, ncol)).copy()
    elif profile == "stairs_up":
        steps = max(3, 4 + level)
        idx = np.clip(((x + 1.0) * 0.5 * steps).astype(int), 0, steps - 1)
        surface = np.broadcast_to((idx / max(steps - 1, 1) - 0.5) * 1.6 * height_span, (nrow, ncol)).copy()
    elif profile == "stairs_down":
        steps = max(3, 4 + level)
        idx = np.clip(((1.0 - (x + 1.0) * 0.5) * steps).astype(int), 0, steps - 1)
        surface = np.broadcast_to((idx / max(steps - 1, 1) - 0.5) * 1.6 * height_span, (nrow, ncol)).copy()
    elif profile == "wall":
        wall_height = 0.08 + 0.02 * level
        wall_x = -0.15 + 0.05 * min(level, 4)
        surface[x[:, 0] > wall_x, :] += wall_height
        surface[x[:, 0] <= wall_x, :] -= 0.25 * wall_height
    elif profile == "blocks":
        surface = _add_blocks(surface, rng, level, height_span)
    else:
        surface = np.broadcast_to(0.55 * height_span * x, (nrow, ncol)).copy()
        surface = _add_blocks(surface, rng, level + 1, 0.7 * height_span)
        surface += _add_wall(surface, x, level)
        surface += 0.05 * height_span * np.sin(4.0 * np.pi * x) * np.cos(3.0 * np.pi * y)

    center_half = 0.12 + 0.02 * min(level, 4)
    center_mask = (np.abs(x) <= center_half) & (np.abs(y) <= center_half)
    surface = surface.copy()
    surface[center_mask] = 0.0
    return surface.astype(np.float32)


def _add_blocks(surface: np.ndarray, rng: np.random.Generator, level: int, height_span: float) -> np.ndarray:
    out = surface.copy()
    nrow, ncol = out.shape
    block_count = 10 + 3 * max(0, level)
    for _ in range(block_count):
        rh = rng.integers(max(3, nrow // 18), max(6, nrow // 8))
        rw = rng.integers(max(3, ncol // 18), max(6, ncol // 8))
        r0 = rng.integers(0, max(1, nrow - rh))
        c0 = rng.integers(0, max(1, ncol - rw))
        height = rng.uniform(-height_span, height_span)
        out[r0:r0 + rh, c0:c0 + rw] += height
    return out


def _add_wall(surface: np.ndarray, x: np.ndarray, level: int) -> np.ndarray:
    out = np.zeros_like(surface)
    wall_height = 0.05 + 0.02 * max(0, level)
    wall_x = -0.1 + 0.05 * min(level, 4)
    out[x[:, 0] > wall_x, :] += wall_height
    out[x[:, 0] <= wall_x, :] -= 0.15 * wall_height
    return out


def _write_hfield(path: Path, surface: np.ndarray) -> None:
    surface = np.asarray(surface, dtype=np.float32)
    min_val = float(np.min(surface))
    max_val = float(np.max(surface))
    scale = max(abs(min_val), abs(max_val), 1e-6)
    normalized = np.clip(surface / (2.0 * scale) + 0.5, 0.0, 1.0).astype(np.float32)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", int(normalized.shape[0]), int(normalized.shape[1])))
        f.write(normalized.tobytes(order="C"))


def _write_scene(scene_path: Path, hfield_name: str, spec: TerrainSpec, surface: np.ndarray) -> None:
    max_val = float(np.max(np.abs(surface)))
    span = max(max_val, 1e-6)
    half_x = spec.length_m * 0.5
    half_y = spec.width_m * 0.5
    include_rel_str = "htdw_4438.xml"
    scene_text = f"""<mujoco model="htdw_4438_{spec.mode}_{spec.profile}_l{spec.level}">
  <include file="{include_rel_str}"/>

  <statistic center="0 0 0.12" extent="0.7"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <hfield name="terrain" file="{hfield_name}" size="{half_x:.6f} {half_y:.6f} {span:.6f} {span:.6f}"/>
  </asset>

  <worldbody>
    <light name="sun" pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <camera name="side" pos="0.55 -0.75 0.35" xyaxes="0.8 0.6 0 -0.18 0.24 0.95"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 0" rgba="0.55 0.58 0.53 1" friction="1.0 0.02 0.001"/>
  </worldbody>
</mujoco>
"""
    scene_path.write_text(scene_text, encoding="utf-8")
