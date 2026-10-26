from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from minidog_rl.envs.minidog_env import CommandRange, MiniDogEnvConfig


TASK_FLAT = "Robot-Flat-v0"
TASK_ROUGH = "Robot-Rough-v0"
TASK_CRAWL = "Robot-Crawl-v0"
LEGACY_TASK = "htdw_4438"


@dataclass(frozen=True)
class TaskEnvCfg:
    task_id: str
    experiment_name: str
    reward_profile: str
    terrain_mode: str
    terrain_profile: str = "mixed"
    terrain_level: int = 0
    terrain_curriculum: bool = False
    terrain_levels: int = 5
    terrain_types: tuple[str, ...] = ("mixed",)
    terrain_width_m: float = 4.0
    terrain_length_m: float = 4.0
    terrain_nrow: int = 128
    terrain_ncol: int = 128
    episode_seconds: float = 8.0
    height: float = 0.235
    terminate_height: float = 0.13
    terminate_rp: float = 0.95
    command_range: CommandRange = field(default_factory=CommandRange)
    action_scale: tuple[float, ...] = (
        0.20, 0.28, 0.28,
        0.20, 0.28, 0.28,
        0.20, 0.28, 0.28,
        0.20, 0.28, 0.28,
    )

    def make_env_config(
        self,
        seed: int,
        fixed_command: tuple[float, float, float] | None = None,
        *,
        terrain_mode: str | None = None,
        terrain_profile: str | None = None,
        terrain_level: int | None = None,
        terrain_curriculum: bool | None = None,
    ) -> MiniDogEnvConfig:
        return MiniDogEnvConfig(
            task_id=self.task_id,
            seed=seed,
            fixed_command=fixed_command,
            episode_seconds=self.episode_seconds,
            height=self.height,
            terminate_height=self.terminate_height,
            terminate_rp=self.terminate_rp,
            command_range=self.command_range,
            action_scale=np.asarray(self.action_scale, dtype=np.float32),
            reward_profile=self.reward_profile,
            terrain_mode=terrain_mode or self.terrain_mode,
            terrain_profile=terrain_profile or self.terrain_profile,
            terrain_level=self.terrain_level if terrain_level is None else terrain_level,
            terrain_curriculum=self.terrain_curriculum if terrain_curriculum is None else terrain_curriculum,
            terrain_levels=self.terrain_levels,
            terrain_types=self.terrain_types,
            terrain_seed=seed,
            terrain_nrow=self.terrain_nrow,
            terrain_ncol=self.terrain_ncol,
            terrain_width_m=self.terrain_width_m,
            terrain_length_m=self.terrain_length_m,
        )


TASKS: dict[str, TaskEnvCfg] = {
    TASK_FLAT: TaskEnvCfg(
        task_id=TASK_FLAT,
        experiment_name="robot_flat",
        reward_profile="flat",
        terrain_mode="plane",
        terrain_profile="flat",
        terrain_curriculum=False,
    ),
    TASK_ROUGH: TaskEnvCfg(
        task_id=TASK_ROUGH,
        experiment_name="robot_rough",
        reward_profile="rough",
        terrain_mode="rough",
        terrain_profile="mixed",
        terrain_curriculum=True,
        terrain_levels=5,
        terrain_types=("slope_up", "slope_down", "stairs_up", "stairs_down", "blocks", "wall", "mixed"),
        episode_seconds=12.0,
    ),
    TASK_CRAWL: TaskEnvCfg(
        task_id=TASK_CRAWL,
        experiment_name="robot_crawl",
        reward_profile="crawl",
        terrain_mode="rough",
        terrain_profile="blocks",
        terrain_curriculum=True,
        terrain_levels=4,
        terrain_types=("blocks", "wall", "mixed"),
        episode_seconds=12.0,
        height=0.175,
        terminate_height=0.10,
        terminate_rp=1.15,
        command_range=CommandRange(vx=(0.0, 0.12), vy=(-0.04, 0.04), yaw_rate=(-0.4, 0.4)),
        action_scale=(
            0.16, 0.24, 0.24,
            0.16, 0.24, 0.24,
            0.16, 0.24, 0.24,
            0.16, 0.24, 0.24,
        ),
    ),
}


ALIASES = {
    LEGACY_TASK: TASK_FLAT,
    "flat": TASK_FLAT,
    "rough": TASK_ROUGH,
    "crawl": TASK_CRAWL,
}


def resolve_task_cfg(task_id: str) -> TaskEnvCfg:
    canonical = ALIASES.get(task_id, task_id)
    if canonical not in TASKS:
        valid = ", ".join([*TASKS.keys(), *ALIASES.keys()])
        raise ValueError(f"Unknown task '{task_id}'. Valid tasks: {valid}")
    return TASKS[canonical]
