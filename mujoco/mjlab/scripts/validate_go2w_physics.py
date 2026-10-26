"""Validate Go2W nominal physics without a policy or PPO updates.

The test keeps the reset pose and actuators nominal, sends zero raw actions,
and stops at the first physical termination.  It is intended to separate an
unstable XML/initial pose/controller from an unstable learned policy.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_warp_cache() -> None:
    try:
        import warp as wp

        cache_dir = ROOT / ".warp_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(cache_dir)
    except Exception:
        pass


def _nominal_cfg(task: str, num_envs: int, seconds: float):
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = max(float(seconds) + 1.0, 5.0)
    cfg.auto_reset = False
    cfg.curriculum = {}

    # Keep only deterministic reset operations.  Startup/reset randomization
    # and pushes would make this a domain-randomization test, not a physics
    # baseline test.
    keep = {"reset_scene", "reset_base", "reset_joints"}
    cfg.events = {name: term for name, term in cfg.events.items() if name in keep}
    if "reset_base" in cfg.events:
        cfg.events["reset_base"].params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        cfg.events["reset_base"].params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    if "reset_joints" in cfg.events:
        cfg.events["reset_joints"].params["position_range"] = (0.0, 0.0)
        cfg.events["reset_joints"].params["velocity_range"] = (0.0, 0.0)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a nominal Go2W zero-action physics test.")
    parser.add_argument("--task", default="Robot-Go2W-Flat-v0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--device", default=None, help="cuda:0, cpu, or omit for automatic selection")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure_warp_cache()
    importlib.import_module("minidog_native")
    importlib.import_module("go2w_native")
    from mjlab.envs import ManagerBasedRlEnv

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = _nominal_cfg(args.task, args.num_envs, args.seconds)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        env.reset(seed=args.seed)
        action_dim = env.action_manager.total_action_dim
        action = torch.zeros((env.num_envs, action_dim), device=env.device)
        max_steps = max(1, math.ceil(args.seconds / env.step_dt))

        robot = env.scene["robot"]
        initial_z = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).detach().clone()
        min_z = initial_z.clone()
        max_tilt = torch.zeros(env.num_envs, device=env.device)
        first_failure_step: int | None = None
        failure_counts: dict[str, int] = {}

        for step in range(1, max_steps + 1):
            _, _, terminated, timeouts, _ = env.step(action)
            z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
            tilt_sin = torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1).clamp(0.0, 1.0)
            tilt_deg = torch.rad2deg(torch.arcsin(tilt_sin))
            min_z = torch.minimum(min_z, z.detach())
            max_tilt = torch.maximum(max_tilt, tilt_deg.detach())

            done = terminated | timeouts
            if bool(done.any()):
                first_failure_step = step
                for name in env.termination_manager.active_terms:
                    count = int(env.termination_manager.get_term(name).sum().item())
                    if count:
                        failure_counts[name] = count
                break

        print(f"task={args.task}")
        print(f"device={env.device}, num_envs={env.num_envs}, action_dim={action_dim}")
        print(f"step_dt={env.step_dt:.4f}, tested_seconds={min(step, max_steps) * env.step_dt:.3f}")
        print(f"initial_height_mean={initial_z.mean().item():.4f}")
        print(f"min_height_min={min_z.min().item():.4f}")
        print(f"max_tilt_deg={max_tilt.max().item():.2f}")
        print(f"first_done_step={first_failure_step if first_failure_step is not None else 'none'}")
        print(f"termination_counts={failure_counts or 'none'}")

        nominal_ok = first_failure_step is None and max_tilt.max().item() < 30.0 and min_z.min().item() > 0.25
        print(f"nominal_physics={'ok' if nominal_ok else 'failed'}")
        return 0 if nominal_ok else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
