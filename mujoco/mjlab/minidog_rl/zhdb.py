from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


ZHDB_ROOT = Path(__file__).resolve().parents[3] / "Zh-db"


class ZhDBClient:
    def __init__(self, run_id: str, name: str, config: dict[str, Any], url: str, enabled: bool = True):
        self.run_id = run_id
        self.name = name
        self.config = config
        self.url = url.rstrip("/")
        self.enabled = enabled

    def log(self, metric: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        payload = {
            "run_id": self.run_id,
            "run": {"name": self.name, "config": self.config, "status": "训练中"},
            "metric": metric,
        }
        return self._post("/api/log", payload)

    def finish(self, status: str = "已完成") -> bool:
        if not self.enabled:
            return False
        return self._post("/api/finish", {"run_id": self.run_id, "status": status})

    def _post(self, path: str, payload: dict[str, Any]) -> bool:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url + path,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return bool(result.get("ok"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"Zh-db log failed: {exc}", file=sys.stderr)
            return False


class ZhDBCallback(BaseCallback):
    def __init__(self, client: ZhDBClient, log_interval_steps: int = 2048, steps_per_iteration: int | None = None):
        super().__init__()
        self.client = client
        self.log_interval_steps = max(1, int(log_interval_steps))
        self.steps_per_iteration = max(1, int(steps_per_iteration or log_interval_steps))
        self.last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step < self.log_interval_steps:
            return True
        self.last_log_step = self.num_timesteps
        metric = self._collect_metric()
        metric["step"] = int(self.num_timesteps)
        metric["sample_step"] = int(self.num_timesteps)
        metric["iteration"] = round(float(self.num_timesteps) / float(self.steps_per_iteration), 6)
        self.client.log(metric)
        return True

    def _on_training_end(self) -> None:
        self.client.finish("已完成")

    def _collect_metric(self) -> dict[str, Any]:
        metric: dict[str, Any] = {}
        name_map = {
            "rollout/ep_rew_mean": "mean_reward",
            "rollout/ep_len_mean": "mean_episode_length",
            "train/value_loss": "loss_value",
            "train/policy_gradient_loss": "loss_surrogate",
            "train/learning_rate": "loss_learning_rate",
            "train/entropy_loss": "loss_entropy",
            "time/fps": "total_fps",
        }
        for src, dst in name_map.items():
            value = self.logger.name_to_value.get(src)
            if isinstance(value, (int, float, np.number)):
                metric[dst] = float(value)

        infos = self.locals.get("infos", [])
        metric.update(self._aggregate_infos(infos))
        return metric

    def _aggregate_infos(self, infos: list[dict[str, Any]]) -> dict[str, float]:
        keys = [
            "error_vel_xy",
            "error_vel_yaw",
            "termination_time_out",
            "termination_nan_detection",
            "termination_bad_orientation",
            "termination_base_ground_contact",
            "terrain_level",
            "reward_track_lin_vel",
            "reward_track_yaw",
            "reward_height",
            "reward_orientation",
            "reward_vertical_velocity",
            "reward_roll_pitch_penalty",
            "reward_terrain_level",
            "reward_action_rate",
            "reward_joint_torques",
            "curriculum_max",
        ]
        values: dict[str, list[float]] = {key: [] for key in keys}
        for info in infos:
            for key in keys:
                value = info.get(key)
                if isinstance(value, (int, float, np.number)):
                    values[key].append(float(value))
        return {key: float(np.mean(items)) for key, items in values.items() if items}
