from __future__ import annotations

import argparse

import torch

from minidog_rl.envs import MiniDogEnvConfig, MiniDogResidualEnv
from minidog_rl.tasks import resolve_task_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a .pt MiniDog policy export.")
    parser.add_argument("--task", type=str, default="Robot-Flat-v0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.09)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--terrain", type=str, default=None, choices=["plane", "rough"])
    parser.add_argument(
        "--terrain-profile",
        "--terrain_profile",
        dest="terrain_profile",
        type=str,
        default=None,
        choices=["mixed", "slope_up", "slope_down", "stairs_up", "stairs_down", "wall", "blocks"],
    )
    parser.add_argument("--terrain-level", "--terrain_level", dest="terrain_level", type=int, default=None)
    parser.add_argument("--terrain-nrow", "--terrain_nrow", dest="terrain_nrow", type=int, default=128)
    parser.add_argument("--terrain-ncol", "--terrain_ncol", dest="terrain_ncol", type=int, default=128)
    parser.add_argument("--terrain-width-m", "--terrain_width_m", dest="terrain_width_m", type=float, default=4.0)
    parser.add_argument("--terrain-length-m", "--terrain_length_m", dest="terrain_length_m", type=float, default=4.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stop-on-done", action="store_true")
    return parser.parse_args()


def load_policy(path: str):
    try:
        package = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        package = torch.load(path, map_location="cpu")
    policy = package["policy"] if isinstance(package, dict) and "policy" in package else package
    policy.set_training_mode(False) if hasattr(policy, "set_training_mode") else policy.eval()
    return policy


def main() -> None:
    args = parse_args()
    task_cfg = resolve_task_cfg(args.task)
    cfg = task_cfg.make_env_config(
        seed=0,
        fixed_command=(args.vx, args.vy, args.yaw),
        terrain_mode=args.terrain,
        terrain_profile=args.terrain_profile,
        terrain_level=args.terrain_level,
        terrain_curriculum=False,
    )
    cfg.terrain_nrow = args.terrain_nrow
    cfg.terrain_ncol = args.terrain_ncol
    cfg.terrain_width_m = args.terrain_width_m
    cfg.terrain_length_m = args.terrain_length_m
    env = MiniDogResidualEnv(cfg, render_mode="human" if args.render else None)
    policy = load_policy(args.model)
    obs, info = env.reset()
    print("step,x,y,z,yaw,body_vx,body_vy,body_wz,reward,fallen")
    for step in range(args.steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 50 == 0 or terminated or truncated:
            body_vel = info["body_vel"]
            body_omega = info["body_omega"]
            x, y = info["base_xy"]
            print(
                f"{step},{x:.4f},{y:.4f},{info['base_z']:.4f},{info['yaw']:.4f},"
                f"{body_vel[0]:.4f},{body_vel[1]:.4f},{body_omega[2]:.4f},"
                f"{reward:.4f},{info['fallen']}"
            )
        if terminated or truncated:
            if args.stop_on_done:
                break
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()
