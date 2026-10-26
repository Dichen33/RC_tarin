from __future__ import annotations

import argparse

import numpy as np

from minidog_rl.envs import MiniDogEnvConfig, MiniDogResidualEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out the MiniDog residual environment and print base logs.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--random-action", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MiniDogEnvConfig(fixed_command=(args.vx, args.vy, args.yaw), seed=args.seed)
    env = MiniDogResidualEnv(cfg, render_mode="human" if args.render else None)
    obs, info = env.reset(seed=args.seed)
    del obs

    print("step,x,y,z,yaw,body_vx,body_vy,body_wz,reward,fallen")
    for step in range(args.steps):
        action = env.action_space.sample() if args.random_action else np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        del obs
        if step % args.log_every == 0 or terminated or truncated:
            body_vel = info["body_vel"]
            body_omega = info["body_omega"]
            x, y = info["base_xy"]
            print(
                f"{step},{x:.4f},{y:.4f},{info['base_z']:.4f},{info['yaw']:.4f},"
                f"{body_vel[0]:.4f},{body_vel[1]:.4f},{body_omega[2]:.4f},"
                f"{reward:.4f},{info['fallen']}"
            )
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
