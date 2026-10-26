from __future__ import annotations

import csv
import math
import os
import statistics
from pathlib import Path
from typing import Any

import torch

from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.utils.logger import Logger


def _scalar(value: Any) -> float | None:
    """Convert a scalar-like tensor/value to a finite float for CSV output."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().float().mean().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# The first unified run collapsed numerically at iteration ~300 and then spent
# 2400 more iterations in a dead regime (action std 141, mean_action_acc 132,
# value loss 1e8, base success rate identically 0).  Aborting a diverged run
# costs one allocation; not aborting costs the whole night.
HEALTH_GUARD_ENV = "MINIDOG_HEALTH_GUARD"
HEALTH_WARMUP_ITERATIONS = 50
HEALTH_ACTION_ACC_LIMIT = 4.0
HEALTH_ACTION_STD_LIMIT = 1.2
HEALTH_VALUE_LOSS_LIMIT = 1.0e5
# A single iteration can spike (a curriculum release changes the return
# distribution) and still recover, so a breach must persist before the run is
# aborted.  plum_step5 tripped a one-iteration 1.6e5 spike and lost 74 minutes of
# training; the collapse it belonged to had already been visible for 100 iterations
# in the mean air time and the fall count.
HEALTH_PERSIST_ITERATIONS = 5


class CsvMetricsLogger(Logger):
    """RSL-RL logger that mirrors iteration summaries into metrics.csv."""

    _base_fields = (
        "iteration",
        "total_steps",
        "mean_reward",
        "mean_episode_length",
        "learning_rate",
        "mean_action_std",
        "loss_value",
        "loss_surrogate",
        "loss_entropy",
        "loss_moe_auxiliary",
        "fps",
        "collection_time",
        "learning_time",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.csv_path = Path(self.log_dir) / "metrics.csv" if self.log_dir else None
        self._csv_fields: list[str] = list(self._base_fields)

    def _extra_values(self) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for info in self.ep_extras:
            for key, value in info.items():
                scalar = _scalar(value)
                if scalar is not None:
                    values.setdefault(str(key), []).append(scalar)
        policy = getattr(self, "policy", None)
        if policy is not None and hasattr(policy, "get_diagnostics"):
            for key, value in policy.get_diagnostics().items():
                scalar = _scalar(value)
                if scalar is not None:
                    values.setdefault(f"Policy/{key}", []).append(scalar)
        return {
            key: statistics.mean(items)
            for key, items in values.items()
            if items
        }

    def _check_health(self, it: int, row: dict[str, float | int | None]) -> None:
        """Abort a run whose policy has gone numerically unstable.

        The guard is intentionally generous: it only has to catch the collapse
        mode where the sampled action amplitude runs away (action std up,
        action acceleration up, value loss exploding) while no reward improves.
        Set ``MINIDOG_HEALTH_GUARD=0`` to disable it.
        """
        flag = os.environ.get(HEALTH_GUARD_ENV, "1").strip().lower()
        if flag in {"0", "false", "no", "off"}:
            return
        if it < HEALTH_WARMUP_ITERATIONS:
            return

        reasons: list[str] = []
        for key, limit in (
            ("mean_action_acc", HEALTH_ACTION_ACC_LIMIT),
            ("mean_action_std", HEALTH_ACTION_STD_LIMIT),
            ("loss_value", HEALTH_VALUE_LOSS_LIMIT),
        ):
            if key not in row:
                continue
            value = row[key]
            if value is None:
                # ``_scalar`` maps NaN/inf to None, so a present-but-None entry
                # means the metric went non-finite.
                reasons.append(f"{key} is not finite")
            elif float(value) > limit:
                reasons.append(f"{key}={float(value):.3g} > {limit:.3g}")

        if not reasons:
            self._health_breach_streak = 0
            return
        # A non-finite metric means the run is already ruined; anything else has to
        # persist for HEALTH_PERSIST_ITERATIONS iterations.
        non_finite = any("not finite" in reason for reason in reasons)
        streak = getattr(self, "_health_breach_streak", 0) + 1
        self._health_breach_streak = streak
        if not non_finite and streak < HEALTH_PERSIST_ITERATIONS:
            return
        raise SystemExit(
            f"[HEALTH GUARD] aborting Go2W training at iteration {it}: "
            + "; ".join(reasons)
            + f" (persisted {streak} iteration(s)). The latest periodic checkpoint "
            "is already on disk; set "
            f"{HEALTH_GUARD_ENV}=0 to disable this check."
        )

    def _write_row(self, row: dict[str, float | int | None]) -> None:
        if self.csv_path is None or self.disable_logs:
            return

        new_fields = [key for key in row if key not in self._csv_fields]
        if new_fields:
            self._csv_fields.extend(sorted(new_fields))
            existing_rows: list[dict[str, str]] = []
            if self.csv_path.exists():
                with self.csv_path.open("r", newline="", encoding="utf-8") as file:
                    existing_rows = list(csv.DictReader(file))
            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self._csv_fields)
                writer.writeheader()
                writer.writerows(existing_rows)

        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self._csv_fields, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        extra_values = self._extra_values()
        collection_size = self.num_envs * self.gpu_world_size * self.cfg["num_steps_per_env"]
        expected_total_steps = self.tot_timesteps + collection_size
        super().log(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=learning_rate,
            action_std=action_std,
            rnd_weight=rnd_weight,
            print_minimal=print_minimal,
            width=width,
            pad=pad,
        )
        if self.writer is not None and not self.disable_logs:
            for key, value in extra_values.items():
                if key.startswith("Policy/"):
                    self.writer.add_scalar(key, value, it)

        row: dict[str, float | int | None] = {
            "iteration": int(it),
            "total_steps": int(max(self.tot_timesteps, expected_total_steps)),
            "mean_reward": statistics.mean(self.rewbuffer) if self.rewbuffer else None,
            "mean_episode_length": statistics.mean(self.lenbuffer) if self.lenbuffer else None,
            "learning_rate": _scalar(learning_rate),
            "mean_action_std": _scalar(action_std),
            "loss_value": _scalar(loss_dict.get("value")),
            "loss_surrogate": _scalar(loss_dict.get("surrogate")),
            "loss_entropy": _scalar(loss_dict.get("entropy")),
            "loss_moe_auxiliary": _scalar(loss_dict.get("moe_auxiliary")),
            "fps": int(collection_size / max(collect_time + learn_time, 1.0e-6)),
            "collection_time": _scalar(collect_time),
            "learning_time": _scalar(learn_time),
        }
        row.update(extra_values)
        self._write_row(row)
        self._check_health(it, row)


class CsvMetricsRunner(MjlabOnPolicyRunner):
    """MJLab runner with a per-iteration CSV beside the TensorBoard event file."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", **kwargs) -> None:
        del kwargs
        super().__init__(env, train_cfg, log_dir, device)
        self.logger = CsvMetricsLogger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        self.logger.policy = self.alg.get_policy()

    def _curriculum_state(self) -> dict[str, Any] | None:
        """Serialize Go2W terrain/LP state alongside the PPO checkpoint."""
        terrain = getattr(getattr(self.env.unwrapped, "scene", None), "terrain", None)
        if terrain is None:
            return None
        names = (
            "_adaptive_initialized",
            # Ratcheted release gates: once a curriculum gate opens it must stay
            # open, including across a stage-transfer resume.
            "_released_gates",
            "_success_ema",
            "_success_count",
            "_episode_counts",
            "_type_level_caps",
            "_type_cap_windows",
            "_course_stage",
            "_lp_probabilities",
            "_lp_reward_ema",
            "_lp_progress",
            "_lp_has_estimate",
            "_lp_stage_sum",
            "_lp_stage_count",
            "_lp_stage_episodes",
            "_lp_stage",
        )
        state: dict[str, Any] = {}
        for name in names:
            value = getattr(terrain, name, None)
            if isinstance(value, torch.Tensor):
                state[name] = value.detach().cpu()
            elif value is not None:
                state[name] = value
        return state or None

    def _restore_curriculum_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        terrain = getattr(getattr(self.env.unwrapped, "scene", None), "terrain", None)
        if terrain is None:
            return
        terrain_origins = getattr(terrain, "terrain_origins", None)
        expected_types = None
        expected_levels = None
        if isinstance(terrain_origins, torch.Tensor) and terrain_origins.ndim >= 2:
            expected_levels, expected_types = terrain_origins.shape[:2]
        type_state_names = {
            "_success_ema",
            "_success_count",
            "_episode_counts",
            "_type_level_caps",
            "_type_cap_windows",
        }
        type_meta_names = {"_adaptive_initialized", "_course_stage"}
        lp_state_names = {
            "_lp_probabilities",
            "_lp_reward_ema",
            "_lp_progress",
            "_lp_has_estimate",
            "_lp_stage_sum",
            "_lp_stage_count",
        }
        lp_meta_names = {"_lp_stage_episodes", "_lp_stage"}
        # Stage-transfer checkpoints can legitimately come from a focused task
        # with fewer terrain columns. Never install tensors with incompatible
        # shapes: the next curriculum callback indexes them by current terrain
        # type and would otherwise fail or silently corrupt statistics.
        type_state_compatible = expected_types is not None
        lp_state_compatible = expected_types is not None and expected_levels is not None
        if type_state_compatible:
            for name in type_state_names:
                value = state.get(name)
                if isinstance(value, torch.Tensor) and tuple(value.shape) != (expected_types,):
                    type_state_compatible = False
                    break
        if lp_state_compatible:
            expected_tasks = expected_levels * expected_types
            for name in lp_state_names:
                value = state.get(name)
                if isinstance(value, torch.Tensor) and tuple(value.shape) != (expected_tasks,):
                    lp_state_compatible = False
                    break
        skipped: list[str] = []
        for name, value in state.items():
            if name in (type_state_names | type_meta_names) and not type_state_compatible:
                skipped.append(name)
                continue
            if name in (lp_state_names | lp_meta_names) and not lp_state_compatible:
                skipped.append(name)
                continue
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
            setattr(terrain, name, value)
        if skipped:
            self._curriculum_restore_skipped = tuple(skipped)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
        """Load normally, or apply an explicit cross-stage transfer policy."""
        mode = os.environ.get("MINIDOG_STAGE_TRANSFER_MODE", "").strip().lower()
        if not mode:
            infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
            self._restore_curriculum_state((infos.get("env_state") or {}).get("go2w_curriculum"))
            return infos
        load_configs = {
            "full_resume": None,
            "actor_critic_reset_optimizer": {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
            "actor_only_reset_critic_optimizer": {"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        }
        if mode not in load_configs:
            raise ValueError(f"Unknown stage transfer mode: {mode}")
        import torch
        source = torch.load(path, map_location="cpu", weights_only=False)
        source_infos = source.get("infos") or {}
        source_env_state = source_infos.get("env_state") or {}
        source_steps = int(source_env_state.get("common_step_counter", 0))
        infos = super().load(path, load_cfg=load_configs[mode], strict=strict, map_location=map_location)
        if mode != "full_resume":
            self.current_learning_iteration = 0
            self.env.unwrapped.common_step_counter = 0
        self._restore_curriculum_state((infos.get("env_state") or {}).get("go2w_curriculum"))
        self._stage_transfer_info = {
            "mode": mode,
            "source_checkpoint": str(path),
            "source_stage": os.environ.get("MINIDOG_STAGE_SOURCE_STAGE", ""),
            "source_global_steps": source_steps,
            "stage_iteration": int(self.current_learning_iteration),
        }
        return infos

    def save(self, path: str, infos=None) -> None:
        stage_info = getattr(self, "_stage_transfer_info", None)
        infos = {**(infos or {})}
        if stage_info:
            infos["stage_transfer"] = stage_info
        env_state = {
            "common_step_counter": self.env.unwrapped.common_step_counter,
        }
        curriculum_state = self._curriculum_state()
        if curriculum_state is not None:
            env_state["go2w_curriculum"] = curriculum_state
        infos["env_state"] = env_state
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)
