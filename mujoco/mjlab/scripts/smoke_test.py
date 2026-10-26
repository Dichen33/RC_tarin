"""Construct and step a small native MJLab environment.

This is intentionally separate from ``native_check.py``: importing and
registering a task does not prove that MuJoCo, Warp, terrain generation,
sensors, or the observation manager can run together.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep first-run Warp compilation inside the portable package. This also makes
# the direct smoke-test command behave like check_linux.sh.
os.environ.setdefault("UV_CACHE_DIR", str(ROOT / ".cache" / "uv"))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))
Path(os.environ["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)


def _is_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_is_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_is_finite(item) for item in value)
    return True


def _shape(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    return type(value).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small native MJLab environment smoke test.")
    parser.add_argument("--task", default="Robot-PlumBlossom-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda:0 when available, otherwise cpu.",
    )
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        raise SystemExit("--num-envs and --steps must be positive")

    import minidog_native  # noqa: F401  # Registers the MiniDog tasks.
    import go2w_native  # noqa: F401  # Registers the Go2W tasks.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42

    env = None
    try:
        print(f"task={args.task}")
        print(f"device={device}")
        print(f"num_envs={args.num_envs}")
        env = ManagerBasedRlEnv(cfg=cfg, device=device)

        observations, _ = env.reset()
        if not isinstance(observations, dict):
            raise RuntimeError(f"reset returned unexpected observations: {type(observations).__name__}")
        required_groups = {"actor_proprio", "actor_radar", "critic"}
        if not required_groups.issubset(observations):
            raise RuntimeError(f"missing observation groups: {sorted(observations)}")
        if not _is_finite(observations):
            raise RuntimeError("reset observations contain NaN or Inf")

        frame_dim = 57 if "Go2W" in args.task else 47
        expected_shapes = {
            "actor_proprio": (env.num_envs, frame_dim * 6),
            "actor_radar": (env.num_envs, 403),
            "critic": (env.num_envs, int(sum(env.observation_manager.group_obs_dim["critic"]))),
        }
        for group_name, expected_shape in expected_shapes.items():
            actual_shape = tuple(observations[group_name].shape)
            if actual_shape != expected_shape:
                raise RuntimeError(f"{group_name} shape={actual_shape}, expected={expected_shape}")

        action_dim = env.action_manager.total_action_dim
        actions = torch.zeros((env.num_envs, action_dim), device=env.device)
        terrain = getattr(env.scene, "terrain", None)
        terrain_names = []
        if terrain is not None and terrain.cfg.terrain_generator is not None:
            terrain_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())

        print(f"actor_proprio={_shape(observations['actor_proprio'])}")
        print(f"actor_radar={_shape(observations['actor_radar'])}")
        print(f"critic_obs={_shape(observations['critic'])}")
        print(f"action_dim={action_dim}")
        print(f"terrain_types={','.join(terrain_names) or 'none'}")

        for step in range(args.steps):
            observations, reward, terminated, truncated, _ = env.step(actions)
            if not _is_finite(observations) or not _is_finite(reward):
                raise RuntimeError(f"step {step + 1} returned NaN or Inf")
            if not _is_finite(terminated) or not _is_finite(truncated):
                raise RuntimeError(f"step {step + 1} returned invalid termination flags")

        reward_mean = float(reward.detach().float().mean().item())
        if not math.isfinite(reward_mean):
            raise RuntimeError("mean reward is not finite")

        # Construct the exact registered PPO runner and execute one sampled
        # action. This catches observation-group, custom-model, and Gaussian
        # distribution contract errors before a long training run.
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=None)
        runner_cls = load_runner_cls(args.task)
        if runner_cls is None:
            raise RuntimeError(f"No runner registered for {args.task}")
        runner = runner_cls(
            wrapped_env,
            asdict(load_rl_cfg(args.task)),
            device=device,
        )
        with torch.inference_mode():
            policy_obs = wrapped_env.get_observations().to(device)
            sampled_action = runner.alg.act(policy_obs)
        if tuple(sampled_action.shape) != (env.num_envs, action_dim):
            raise RuntimeError(f"sampled action shape={tuple(sampled_action.shape)}")
        if not _is_finite(sampled_action):
            raise RuntimeError("sampled action contains NaN or Inf")
        print(f"sampled_action={_shape(sampled_action)}")
        print(f"steps_ok={args.steps}")
        print(f"mean_reward={reward_mean:.6f}")
        print("smoke_test=ok")
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
