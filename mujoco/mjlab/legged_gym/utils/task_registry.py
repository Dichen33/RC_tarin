from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    env_cls: type
    env_cfg: type
    train_cfg: type


class TaskRegistry:
    def __init__(self) -> None:
        self.task_classes: dict[str, TaskSpec] = {}

    def register(self, name: str, env_cls: type, env_cfg: type, train_cfg: type) -> None:
        self.task_classes[name] = TaskSpec(env_cls=env_cls, env_cfg=env_cfg, train_cfg=train_cfg)

    def get_cfgs(self, name: str) -> tuple[type, type]:
        spec = self._get(name)
        return spec.env_cfg, spec.train_cfg

    def make_env(self, name: str, args: Any | None = None):
        from minidog_rl.tasks import resolve_task_cfg

        spec = self._get(name)
        seed = getattr(args, "seed", 1) if args is not None else 1
        task_cfg = resolve_task_cfg(name)
        return spec.env_cls(task_cfg.make_env_config(seed=seed)), spec.env_cfg

    def _get(self, name: str) -> TaskSpec:
        if name not in self.task_classes:
            raise KeyError(f"Unknown task: {name}")
        return self.task_classes[name]


task_registry = TaskRegistry()
