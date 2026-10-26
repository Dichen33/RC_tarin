from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from minidog_rl.envs import MiniDogEnvConfig, MiniDogResidualEnv
from minidog_rl.paths import MODEL_ROOT, RUNS_ROOT
from minidog_rl.tasks import TaskEnvCfg, resolve_task_cfg
from minidog_rl.zhdb import ZhDBCallback, ZhDBClient


def make_env(
    seed: int,
    task_cfg: TaskEnvCfg,
    fixed_command: tuple[float, float, float] | None = None,
    terrain_mode: str | None = None,
    terrain_profile: str | None = None,
    terrain_level: int | None = None,
    terrain_curriculum: bool | None = None,
    terrain_nrow: int = 128,
    terrain_ncol: int = 128,
    terrain_width_m: float = 4.0,
    terrain_length_m: float = 4.0,
):
    def _factory():
        cfg = task_cfg.make_env_config(
            seed=seed,
            fixed_command=fixed_command,
            terrain_mode=terrain_mode,
            terrain_profile=terrain_profile,
            terrain_level=terrain_level,
            terrain_curriculum=terrain_curriculum,
        )
        cfg.terrain_seed = seed
        cfg.terrain_nrow = terrain_nrow
        cfg.terrain_ncol = terrain_ncol
        cfg.terrain_width_m = terrain_width_m
        cfg.terrain_length_m = terrain_length_m
        return Monitor(MiniDogResidualEnv(cfg))

    return _factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO residual policy for HTDW-4438.")
    parser.add_argument("--task", type=str, default="Robot-Flat-v0")
    parser.add_argument("--headless", action="store_true", help="Accepted for IsaacGym-style command compatibility.")
    parser.add_argument("--resume", action="store_true", help="Resume from a saved SB3 model.")
    parser.add_argument("--load-run", "--load_run", dest="load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="final_model")
    parser.add_argument("--sim-device", "--sim_device", dest="sim_device", type=str, default=None, help="Accepted for IsaacGym compatibility.")
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=4)
    parser.add_argument(
        "--iterations",
        "--num-iterations",
        "--max-iterations",
        "--max_iterations",
        dest="iterations",
        type=int,
        default=1000,
    )
    parser.add_argument("--steps-per-env", "--num-steps-per-env", "--num_steps_per_env", dest="steps_per_env", type=int, default=24)
    parser.add_argument("--total-steps", type=int, default=None, help="Compatibility option. Prefer --iterations.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--experiment-name", "--experiment_name", dest="experiment_name", type=str, default=None)
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default="residual_v1")
    parser.add_argument("--device", "--rl_device", dest="device", type=str, default="auto")
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--num-mini-batches", "--num_mini_batches", dest="num_mini_batches", type=int, default=4)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--epochs", "--num-learning-epochs", "--num_learning_epochs", dest="epochs", type=int, default=5)
    parser.add_argument("--save-interval", "--save_interval", dest="save_interval", type=int, default=200, help="Checkpoint interval in PPO iterations.")
    parser.add_argument("--eval-interval", "--eval_interval", dest="eval_interval", type=int, default=200, help="Evaluation interval in PPO iterations.")
    parser.add_argument("--eval-vx", type=float, default=0.0)
    parser.add_argument("--eval-vy", type=float, default=0.09)
    parser.add_argument("--eval-yaw", type=float, default=0.0)
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
    parser.add_argument("--terrain-curriculum", "--terrain_curriculum", dest="terrain_curriculum", action="store_true", default=None)
    parser.add_argument("--no-terrain-curriculum", dest="terrain_curriculum", action="store_false")
    parser.add_argument("--terrain-nrow", "--terrain_nrow", dest="terrain_nrow", type=int, default=128)
    parser.add_argument("--terrain-ncol", "--terrain_ncol", dest="terrain_ncol", type=int, default=128)
    parser.add_argument("--terrain-width-m", "--terrain_width_m", dest="terrain_width_m", type=float, default=4.0)
    parser.add_argument("--terrain-length-m", "--terrain_length_m", dest="terrain_length_m", type=float, default=4.0)
    parser.add_argument("--zhdb-url", type=str, default="http://127.0.0.1:8765")
    parser.add_argument("--no-zhdb", action="store_true", help="Disable live logging to Zh-db.")
    parser.add_argument("--zhdb-log-interval", type=int, default=None)
    return parser.parse_args()


def resolve_training_schedule(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1.")
    if args.steps_per_env < 1:
        raise ValueError("--steps-per-env must be >= 1.")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1.")
    if args.num_mini_batches < 1:
        raise ValueError("--num-mini-batches must be >= 1.")

    rollout_size = args.num_envs * args.steps_per_env
    if rollout_size <= 1:
        raise ValueError("PPO requires num_envs * steps_per_env > 1.")

    if args.total_steps is None:
        total_timesteps = rollout_size * args.iterations
        iterations = args.iterations
    else:
        if args.total_steps < rollout_size:
            raise ValueError("--total-steps must be >= num_envs * steps_per_env.")
        total_timesteps = args.total_steps
        iterations = math.ceil(total_timesteps / rollout_size)

    return rollout_size, total_timesteps, iterations


def resolve_batch_size(args: argparse.Namespace, rollout_size: int) -> int:
    if args.batch_size is None:
        batch_size = rollout_size // args.num_mini_batches
    else:
        batch_size = args.batch_size
    return max(2, min(batch_size, rollout_size))


def resolve_checkpoint_path(args: argparse.Namespace, experiment_name: str) -> Path:
    load_run = args.load_run or args.run_name
    checkpoint = str(args.checkpoint)
    candidates: list[Path] = []

    if checkpoint.endswith(".zip"):
        candidates.append(RUNS_ROOT / experiment_name / load_run / checkpoint)
    else:
        candidates.extend(
            [
                RUNS_ROOT / experiment_name / load_run / f"{checkpoint}.zip",
                RUNS_ROOT / experiment_name / load_run / f"checkpoint_{checkpoint}_steps.zip",
                RUNS_ROOT / experiment_name / load_run / f"model_{checkpoint}.zip",
                MODEL_ROOT / experiment_name / load_run / f"{checkpoint}.zip",
                MODEL_ROOT / experiment_name / load_run / f"checkpoint_{checkpoint}_steps.zip",
                MODEL_ROOT / experiment_name / load_run / f"model_{checkpoint}.zip",
            ]
        )

    legacy_root = RUNS_ROOT.parent / "runs"
    legacy_candidates = []
    for candidate in candidates:
        try:
            legacy_candidates.append(legacy_root / candidate.relative_to(RUNS_ROOT))
        except ValueError:
            pass
    for path in [*candidates, *legacy_candidates]:
        if path.exists():
            return path
    return candidates[0]


def save_playable_policy_pt(model: PPO, path: Path, metadata: dict[str, object]) -> None:
    model.policy.to("cpu")
    model.policy.set_training_mode(False)
    torch.save(
        {
            "format": "stable_baselines3_policy",
            "policy": model.policy,
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    task_cfg = resolve_task_cfg(args.task)
    experiment_name = args.experiment_name or task_cfg.experiment_name
    rollout_size, total_timesteps, resolved_iterations = resolve_training_schedule(args)
    batch_size = resolve_batch_size(args, rollout_size)

    run_dir = RUNS_ROOT / experiment_name / args.run_name
    model_dir = MODEL_ROOT / experiment_name / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = str(run_dir / "tb") if importlib.util.find_spec("tensorboard") is not None else None
    if tensorboard_log is None:
        print("tensorboard is not installed; continuing without tensorboard logging.")
    progress_bar = importlib.util.find_spec("tqdm") is not None and importlib.util.find_spec("rich") is not None
    if not progress_bar:
        print("tqdm/rich are not installed; continuing without SB3 progress bar.")

    env_fns = [
        make_env(
            args.seed + i,
            task_cfg,
            terrain_mode=args.terrain,
            terrain_profile=args.terrain_profile,
            terrain_level=args.terrain_level,
            terrain_curriculum=args.terrain_curriculum,
            terrain_nrow=args.terrain_nrow,
            terrain_ncol=args.terrain_ncol,
            terrain_width_m=args.terrain_width_m,
            terrain_length_m=args.terrain_length_m,
        )
        for i in range(args.num_envs)
    ]
    vec_env_cls = DummyVecEnv if args.num_envs == 1 else SubprocVecEnv
    env = vec_env_cls(env_fns)
    eval_env = DummyVecEnv(
        [
            make_env(
                args.seed + 10_000,
                task_cfg,
                (args.eval_vx, args.eval_vy, args.eval_yaw),
                terrain_mode=args.terrain,
                terrain_profile=args.terrain_profile,
                terrain_level=args.terrain_level,
                terrain_curriculum=False,
                terrain_nrow=args.terrain_nrow,
                terrain_ncol=args.terrain_ncol,
                terrain_width_m=args.terrain_width_m,
                terrain_length_m=args.terrain_length_m,
            )
        ]
    )

    if args.resume:
        checkpoint_path = resolve_checkpoint_path(args, experiment_name)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        print(f"Resuming from {checkpoint_path}")
        model = PPO.load(str(checkpoint_path), env=env, device=args.device, tensorboard_log=tensorboard_log)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.steps_per_env,
            batch_size=batch_size,
            n_epochs=args.epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            device=args.device,
        )

    save_freq = max(1, args.save_interval * args.steps_per_env)
    eval_freq = max(1, args.eval_interval * args.steps_per_env)
    callbacks = [
        CheckpointCallback(save_freq=save_freq, save_path=str(model_dir), name_prefix="checkpoint"),
        EvalCallback(eval_env, best_model_save_path=str(model_dir), log_path=str(run_dir / "eval"), eval_freq=eval_freq),
    ]
    zhdb_client = ZhDBClient(
        run_id=args.run_name,
        name=f"MJLab训练：{args.run_name}",
        config={
            "算法": "PPO",
            "环境": "MiniDogResidualEnv",
            "控制形式": "IK gait + 12 关节残差",
            "task": args.task,
            "canonical_task": task_cfg.task_id,
            "experiment_name": experiment_name,
            "num_envs": args.num_envs,
            "steps_per_env": args.steps_per_env,
            "iterations": resolved_iterations,
            "rollout_size": rollout_size,
            "num_mini_batches": args.num_mini_batches,
            "total_timesteps": total_timesteps,
            "batch_size": batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "seed": args.seed,
            "eval_command": [args.eval_vx, args.eval_vy, args.eval_yaw],
            "terrain": args.terrain or task_cfg.terrain_mode,
            "terrain_profile": args.terrain_profile or task_cfg.terrain_profile,
            "terrain_level": task_cfg.terrain_level if args.terrain_level is None else args.terrain_level,
            "terrain_curriculum": task_cfg.terrain_curriculum if args.terrain_curriculum is None else args.terrain_curriculum,
            "terrain_types": list(task_cfg.terrain_types),
            "terrain_levels": task_cfg.terrain_levels,
        },
        url=args.zhdb_url,
        enabled=not args.no_zhdb,
    )
    zhdb_log_interval = args.zhdb_log_interval or rollout_size
    callbacks.append(
        ZhDBCallback(
            zhdb_client,
            log_interval_steps=zhdb_log_interval,
            steps_per_iteration=rollout_size,
        )
    )

    print(
        "Training schedule: "
        f"num_envs={args.num_envs}, steps_per_env={args.steps_per_env}, "
        f"iterations={resolved_iterations}, total_timesteps={total_timesteps}, batch_size={batch_size}"
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=progress_bar)
    export_metadata = {
        "task": args.task,
        "canonical_task": task_cfg.task_id,
        "experiment_name": experiment_name,
        "run_name": args.run_name,
        "backend": "stable-baselines3",
        "num_envs": args.num_envs,
        "steps_per_env": args.steps_per_env,
        "iterations": resolved_iterations,
        "total_timesteps": total_timesteps,
        "terrain": args.terrain or task_cfg.terrain_mode,
        "terrain_profile": args.terrain_profile or task_cfg.terrain_profile,
        "terrain_level": task_cfg.terrain_level if args.terrain_level is None else args.terrain_level,
        "terrain_curriculum": task_cfg.terrain_curriculum if args.terrain_curriculum is None else args.terrain_curriculum,
    }
    model.save(str(model_dir / "final_model.zip"))
    save_playable_policy_pt(model, model_dir / "final_policy.pt", export_metadata)
    env.close()
    eval_env.close()
    print(f"Saved training artifacts to {Path(run_dir).resolve()}")
    print(f"Saved SB3 model to {Path(model_dir / 'final_model.zip').resolve()}")
    print(f"Saved playable policy to {Path(model_dir / 'final_policy.pt').resolve()}")


if __name__ == "__main__":
    main()
