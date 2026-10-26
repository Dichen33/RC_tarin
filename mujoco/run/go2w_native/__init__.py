from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from .config.env_cfgs import (
    go2w_gap_env_cfg,
    go2w_flat_env_cfg,
    go2w_high_wall_env_cfg,
    go2w_mixed_env_cfg,
    go2w_plum_blossom_env_cfg,
    go2w_rough_env_cfg,
    go2w_step_env_cfg,
)
from .config.rl_cfg import go2w_ppo_runner_cfg
from .runner import CsvMetricsRunner


def _register(task_id: str, env_cfg, run_name: str, residual_scale: float) -> None:
    try:
        train_cfg = env_cfg()
        play_cfg = env_cfg(play=True)
    except ValueError as exc:
        # A focused task whose obstacle family was switched off through
        # MINIDOG_OBSTACLES must not abort the import of every other task.
        print(f"[Go2W] task {task_id} unavailable: {exc}", flush=True)
        return
    register_mjlab_task(
        task_id=task_id,
        env_cfg=train_cfg,
        play_env_cfg=play_cfg,
        rl_cfg=go2w_ppo_runner_cfg(run_name, residual_scale=residual_scale),
        runner_cls=CsvMetricsRunner,
    )


_register("Robot-Go2W-Flat-v0", go2w_flat_env_cfg, "go2w_flat", residual_scale=0.0)
_register("Robot-Go2W-Rough-v0", go2w_rough_env_cfg, "go2w_rough", residual_scale=0.0)
_register("Robot-Go2W-Step-v0", go2w_step_env_cfg, "go2w_step", residual_scale=0.20)
_register("Robot-Go2W-PlumBlossom-v0", go2w_plum_blossom_env_cfg, "go2w_plum_blossom", residual_scale=0.35)
_register("Robot-Go2W-Gap-v0", go2w_gap_env_cfg, "go2w_gap", residual_scale=0.30)
_register("Robot-Go2W-HighWall-v0", go2w_high_wall_env_cfg, "go2w_high_wall", residual_scale=0.40)
_register("Robot-Go2W-Mixed-v0", go2w_mixed_env_cfg, "go2w_mixed", residual_scale=0.40)

