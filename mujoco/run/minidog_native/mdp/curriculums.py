from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..config.course_profiles import (
    BASE_TERRAIN_NAMES,
    SPECIAL_OBSTACLE_NAMES,
    STANDARD_OBSTACLE_NAMES,
)


# ETH-style terrain curricula use a local success signal to move an episode up
# one row and a failure signal to move it down one row.  The additional cap
# below is per terrain type: a wall can remain at its low row while the flat
# and stair policies continue improving.
EPISODES_PER_DIFFICULTY_UPGRADE = 256
DIFFICULTY_UPGRADE_SUCCESS = 0.75
DIFFICULTY_DOWNGRADE_SUCCESS = 0.45

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    success_distance: float | None = None,
    fail_distance_ratio: float = 0.35,
) -> dict[str, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
        return {}

    command = env.command_manager.get_command(command_name)
    displacement = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    if success_distance is None:
        distance = torch.norm(displacement, dim=1)
        terrain_length = terrain.cfg.terrain_generator.size[0]
        move_up = distance > 0.5 * terrain_length
    else:
        # Plum-blossom courses have a short usable route inside a larger patch.
        # Use forward displacement to the exit platform instead of the patch
        # midpoint, otherwise successful episodes rarely advance the curriculum.
        distance = torch.clamp(displacement[:, 0], min=0.0)
        move_up = displacement[:, 0] > success_distance

    terrain_types = getattr(terrain, "terrain_types", None)
    obstacle_success = getattr(env, "_obstacle_course_success", None)
    special_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    if terrain_types is not None and obstacle_success is not None:
        episode_types = terrain_types[env_ids]
        special_names = {"flat_gap", "rough_plum_blossom", "flat_high_wall"}
        sub_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        special_ids = [sub_names.index(name) for name in special_names if name in sub_names]
        if special_ids:
            special_mask = torch.isin(
                episode_types,
                torch.tensor(special_ids, dtype=torch.long, device=env.device),
            )
            # Focused obstacle tasks and the mixed course use the event
            # detector from course_completion instead of displacement alone.
            move_up = torch.where(special_mask, obstacle_success[env_ids], move_up)

    command_distance = torch.abs(command[env_ids, 0]) * env.max_episode_length_s
    move_down = (distance < fail_distance_ratio * command_distance) & ~move_up
    terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
    timeouts = getattr(getattr(env, "termination_manager", None), "time_outs", None)
    if terminated is not None or timeouts is not None:
        failed_episode = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        if terminated is not None:
            failed_episode |= terminated[env_ids]
        if timeouts is not None:
            failed_episode |= timeouts[env_ids]
        # A fall or timeout before the stable exit is an explicit downgrade
        # signal even when the robot covered more than the distance ratio.
        move_down |= special_mask & ~move_up & failed_episode
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()

    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "episode_success_rate_batch": torch.mean(move_up.float()),
    }

    origins = getattr(terrain, "terrain_origins", None)
    if terrain_types is not None and origins is not None:
        sub_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        episode_types = terrain_types[env_ids]
        if origins.shape[1] == len(sub_names):
            for index, name in enumerate(sub_names):
                mask = terrain_types == index
                if mask.any():
                    result[name] = torch.mean(levels[mask])
                episode_mask = episode_types == index
                if episode_mask.any():
                    result[f"{name}_success_rate_batch"] = torch.mean(move_up[episode_mask].float())
    return result


def terrain_levels_focused_obstacle(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    success_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fail_distance_ratio: float = 0.35,
    primary_terrain_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Promote a single-obstacle task only after its stable exit event."""
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
        return {}
    command = env.command_manager.get_command(command_name)
    displacement = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    distance = torch.clamp(displacement[:, 0], min=0.0)
    terrain_types = getattr(terrain, "terrain_types", None)
    sub_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    episode_types = terrain_types[env_ids] if terrain_types is not None else None
    primary_mask = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    if primary_terrain_name is not None and episode_types is not None and primary_terrain_name in sub_names:
        primary_mask = episode_types == sub_names.index(primary_terrain_name)
    obstacle_success = getattr(env, "_obstacle_course_success", None)
    if obstacle_success is not None:
        # Only the focused obstacle is gated by its stable completion event.
        # History/revisit terrain uses ordinary progress and therefore cannot
        # inflate the primary obstacle success statistic.
        move_up = torch.where(
            primary_mask,
            obstacle_success[env_ids],
            _valid_course_exit(env, env_ids, displacement[:, 0] > success_distance),
        )
    else:
        move_up = _valid_course_exit(env, env_ids, displacement[:, 0] > success_distance)
    command_distance = torch.abs(command[env_ids, 0]) * env.max_episode_length_s
    move_down = (distance < fail_distance_ratio * command_distance) & ~move_up
    terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
    timeouts = getattr(getattr(env, "termination_manager", None), "time_outs", None)
    if terminated is not None or timeouts is not None:
        failed_episode = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        if terminated is not None:
            failed_episode |= terminated[env_ids]
        if timeouts is not None:
            failed_episode |= timeouts[env_ids]
        move_down |= ~move_up & failed_episode
    terrain.update_env_origins(env_ids, move_up, move_down)
    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "episode_success_rate_batch": torch.mean(move_up.float()),
    }
    if terrain_types is not None:
        names = sub_names
        for index, name in enumerate(names):
            mask = terrain_types == index
            if mask.any():
                result[name] = torch.mean(levels[mask])
            episode_mask = episode_types == index
            if episode_mask.any():
                result[f"{name}_success_rate_batch"] = torch.mean(move_up[episode_mask].float())
        if primary_mask.any():
            result["primary_obstacle_success_rate_batch"] = torch.mean(move_up[primary_mask].float())
        if (~primary_mask).any():
            result["history_revisit_success_rate_batch"] = torch.mean(move_up[~primary_mask].float())
    return result


def _valid_course_exit(env: ManagerBasedRlEnv, env_ids: torch.Tensor, move_up: torch.Tensor) -> torch.Tensor:
    """Filter curriculum progress to upright, non-failed episode exits.

    ``_reset_idx`` runs the curriculum *before* the termination manager clears
    its buffers, so these flags still describe the episode that just ended.
    A timeout is the normal successful ending and must not disqualify an
    episode; only genuine failure terminations do.
    """
    asset: Entity = env.scene["robot"]
    relative_height = asset.data.root_link_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
    tilt = torch.linalg.vector_norm(asset.data.projected_gravity_b[env_ids, :2], dim=1)
    terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
    if terminated is None:
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (
        move_up
        & (relative_height > 0.14)
        & (tilt < 0.70)
        & ~terminated[env_ids]
    )


def terrain_levels_plum_obstacle_release(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    success_distance: float = 2.7,
    gap_success_distance: float = 3.35,
    fail_distance_ratio: float = 0.35,
    wall_success_distance: float = 4.80,
    default_success_distance: float = 4.0,
    base_terrain_names: tuple[str, ...] = BASE_TERRAIN_NAMES,
    standard_obstacle_names: tuple[str, ...] = STANDARD_OBSTACLE_NAMES,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Adaptive curriculum with a hard basic-locomotion gate.

    Obstacles are not released by wall-clock training steps. Environments
    start on flat/shallow rough terrain, and each obstacle family is released
    only after the previous family reaches a measured success threshold.
    Per-type exponential success statistics are kept on the terrain entity so
    every reset can raise or lower its own difficulty level.
    """
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
        return {}
    if terrain.terrain_origins is None or terrain.env_origins is None:
        return {}

    generator = terrain.cfg.terrain_generator
    sub_names = list(generator.sub_terrains.keys())
    required_names = ("flat_gap", "rough_plum_blossom", "flat_high_wall")
    if any(name not in sub_names for name in required_names):
        return terrain_levels_velocity(
            env,
            env_ids,
            command_name,
            asset_cfg,
            success_distance=success_distance,
            fail_distance_ratio=fail_distance_ratio,
        )

    # Evaluate the episode that just ended before changing the terrain type.
    # This is intentionally done before RewardManager.reset() clears the
    # per-episode plum-blossom foothold state.
    command = env.command_manager.get_command(command_name)
    displacement = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    distance = torch.clamp(displacement[:, 0], min=0.0)
    target_distance = torch.full_like(distance, default_success_distance)
    base_ids = [sub_names.index(name) for name in base_terrain_names if name in sub_names]
    previous_type_ids = terrain.terrain_types[env_ids]
    if base_ids:
        target_distance = torch.where(
            torch.isin(previous_type_ids, torch.tensor(base_ids, device=env.device)),
            torch.full_like(target_distance, 2.5),
            target_distance,
        )
    if "rough_plum_blossom" in sub_names:
        target_distance = torch.where(
            previous_type_ids == sub_names.index("rough_plum_blossom"),
            torch.full_like(target_distance, success_distance),
            target_distance,
        )
    if "flat_gap" in sub_names:
        target_distance = torch.where(
            previous_type_ids == sub_names.index("flat_gap"),
            torch.full_like(target_distance, gap_success_distance),
            target_distance,
        )
    if "flat_high_wall" in sub_names:
        target_distance = torch.where(
            previous_type_ids == sub_names.index("flat_high_wall"),
            torch.full_like(target_distance, wall_success_distance),
            target_distance,
        )
    move_up = _valid_course_exit(env, env_ids, displacement[:, 0] > target_distance)
    previous_names = torch.tensor(
        [sub_names.index("rough_plum_blossom")] if "rough_plum_blossom" in sub_names else [],
        device=env.device,
        dtype=torch.long,
    )
    if len(previous_names) > 0:
        plum_mask = previous_type_ids == previous_names[0]
        plum_success = getattr(env, "_plum_course_success", None)
        if plum_success is not None:
            move_up = torch.where(plum_mask, plum_success[env_ids], move_up)
    obstacle_success = getattr(env, "_obstacle_course_success", None)
    special_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    if obstacle_success is not None:
        special_ids = [
            sub_names.index(name)
            for name in ("flat_gap", "flat_high_wall")
            if name in sub_names
        ]
        if special_ids:
            special_mask = torch.isin(
                previous_type_ids,
                torch.tensor(special_ids, dtype=torch.long, device=env.device),
            )
            move_up = torch.where(special_mask, obstacle_success[env_ids], move_up)

    if not hasattr(terrain, "_adaptive_initialized"):
        n_types = len(sub_names)
        terrain._adaptive_initialized = False
        terrain._success_ema = torch.zeros(n_types, device=env.device)
        terrain._success_count = torch.zeros(n_types, device=env.device)
        terrain._episode_counts = torch.zeros(n_types, device=env.device)
        # All newly released terrain types begin at row 0/1.  Their cap is
        # raised only after a complete success window, up to the final expert
        # row generated from the XML medium anchor.
        terrain._type_level_caps = torch.ones(n_types, dtype=torch.long, device=env.device)
        terrain._type_cap_windows = torch.zeros(n_types, dtype=torch.long, device=env.device)
        terrain._course_stage = 0

    # The first reset has no completed episode to score. Do not contaminate
    # the statistics with the random terrain assignment made at construction.
    if terrain._adaptive_initialized:
        for type_id in range(len(sub_names)):
            mask = previous_type_ids == type_id
            if mask.any():
                rate = torch.mean(move_up[mask].float())
                terrain._success_ema[type_id] = 0.8 * terrain._success_ema[type_id] + 0.2 * rate
                episode_count = torch.count_nonzero(mask).float()
                terrain._success_count[type_id] += move_up[mask].float().sum()
                terrain._episode_counts[type_id] += episode_count
    else:
        terrain._adaptive_initialized = True

    def aggregate(names: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [sub_names.index(name) for name in names if name in sub_names]
        if not ids:
            zero = torch.zeros((), device=env.device)
            return zero, zero
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=env.device)
        counts = terrain._episode_counts[ids_tensor].sum()
        successes = terrain._success_count[ids_tensor].sum()
        return successes / torch.clamp(counts, min=1.0), counts

    def aggregate_ema(names: tuple[str, ...]) -> torch.Tensor:
        ids = [sub_names.index(name) for name in names if name in sub_names]
        if not ids:
            return torch.zeros((), device=env.device)
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=env.device)
        counts = terrain._episode_counts[ids_tensor].sum()
        weighted = (terrain._success_ema[ids_tensor] * terrain._episode_counts[ids_tensor]).sum()
        return weighted / torch.clamp(counts, min=1.0)

    base_rate, base_count = aggregate(base_terrain_names)
    standard_rate, standard_count = aggregate(standard_obstacle_names)
    gap_rate, gap_count = aggregate(("flat_gap",))
    plum_rate, plum_count = aggregate(("rough_plum_blossom",))
    wall_rate, wall_count = aggregate(("flat_high_wall",))

    # Advance each terrain type's difficulty cap once per completed window.
    # This is deliberately based on cumulative success, so one lucky batch
    # cannot release a high wall or a deep gap immediately.
    max_generated_level = terrain.terrain_origins.shape[0] - 1
    type_rates = terrain._success_count / torch.clamp(terrain._episode_counts, min=1.0)
    completed_windows = torch.floor(
        terrain._episode_counts / float(EPISODES_PER_DIFFICULTY_UPGRADE)
    ).to(torch.long)
    new_windows = completed_windows > terrain._type_cap_windows
    can_upgrade = new_windows & (type_rates >= DIFFICULTY_UPGRADE_SUCCESS)
    should_downgrade = new_windows & (
        completed_windows >= 2
    ) & (type_rates < DIFFICULTY_DOWNGRADE_SUCCESS)
    caps = terrain._type_level_caps
    caps = torch.where(can_upgrade, torch.clamp(caps + 1, max=max_generated_level), caps)
    caps = torch.where(should_downgrade, torch.clamp(caps - 1, min=1), caps)
    terrain._type_level_caps = caps
    terrain._type_cap_windows = torch.maximum(terrain._type_cap_windows, completed_windows)

    # Each gate needs enough completed episodes to avoid releasing an
    # obstacle because of one lucky rollout.
    base_ready = bool(base_count.item() >= 2048 and base_rate.item() >= 0.75)
    standard_ready = bool(standard_count.item() >= 1024 and standard_rate.item() >= 0.65)
    gap_ready = bool(gap_count.item() >= 512 and gap_rate.item() >= 0.60)
    plum_ready = bool(plum_count.item() >= 512 and plum_rate.item() >= 0.60)
    wall_ready = bool(wall_count.item() >= 512 and wall_rate.item() >= 0.60)

    # Keep the stage table explicit.  This mirrors the reference project's
    # terrain profile ordering and makes it difficult to accidentally release
    # a later obstacle when changing one threshold.
    stage_profiles = (
        (base_terrain_names, 1),
        (base_terrain_names + standard_obstacle_names, 3),
        (base_terrain_names + standard_obstacle_names + ("flat_gap",), 4),
        (base_terrain_names + standard_obstacle_names + ("flat_gap", "rough_plum_blossom"), 6),
        (base_terrain_names + standard_obstacle_names + SPECIAL_OBSTACLE_NAMES, 8),
        (base_terrain_names + standard_obstacle_names + SPECIAL_OBSTACLE_NAMES, terrain.terrain_origins.shape[0] - 1),
    )
    gates = (base_ready, standard_ready, gap_ready, plum_ready, wall_ready)
    stage = next((index for index, ready in enumerate(gates) if not ready), len(gates))
    allowed_names, max_level = stage_profiles[stage]

    max_level = min(max_level, terrain.terrain_origins.shape[0] - 1)
    allowed_names = tuple(dict.fromkeys(name for name in allowed_names if name in sub_names))
    allowed_ids = torch.tensor([sub_names.index(name) for name in allowed_names], dtype=torch.long, device=env.device)
    terrain._course_stage = stage

    # The reference task also limits command ranges on difficult terrain.  A
    # small, stage-wise command schedule keeps early locomotion learnable and
    # prevents a newly released obstacle from being paired with a harder
    # command distribution on the same reset.
    command_ranges = (
        ((0.10, 0.22), (-0.02, 0.02), (-0.10, 0.10)),
        ((0.12, 0.26), (-0.03, 0.03), (-0.12, 0.12)),
        ((0.14, 0.28), (-0.04, 0.04), (-0.15, 0.15)),
        ((0.16, 0.32), (-0.04, 0.04), (-0.18, 0.18)),
        ((0.18, 0.35), (-0.05, 0.05), (-0.20, 0.20)),
        ((0.18, 0.40), (-0.08, 0.08), (-0.25, 0.25)),
    )
    try:
        command_cfg = env.command_manager.get_term_cfg(command_name)
        command_cfg.ranges.lin_vel_x = command_ranges[stage][0]
        command_cfg.ranges.lin_vel_y = command_ranges[stage][1]
        command_cfg.ranges.ang_vel_z = command_ranges[stage][2]
    except (AttributeError, KeyError):
        # Keep the curriculum usable with environments that do not use a
        # UniformVelocityCommand term.
        pass

    command_distance = torch.abs(command[env_ids, 0]) * env.max_episode_length_s
    move_down = (distance < fail_distance_ratio * command_distance) & ~move_up
    terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
    timeouts = getattr(getattr(env, "termination_manager", None), "time_outs", None)
    if terminated is not None or timeouts is not None:
        failed_episode = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        if terminated is not None:
            failed_episode |= terminated[env_ids]
        if timeouts is not None:
            failed_episode |= timeouts[env_ids]
        move_down |= special_mask & ~move_up & failed_episode
    terrain.terrain_levels[env_ids] += move_up.to(torch.long) - move_down.to(torch.long)
    type_caps = terrain._type_level_caps[terrain.terrain_types[env_ids]]
    terrain.terrain_levels[env_ids] = torch.minimum(
        torch.clamp(terrain.terrain_levels[env_ids], min=0, max=max_level), type_caps
    )

    # Existing environments may have been initialized on a later column.
    # Resample them into the currently released set at the next reset.
    current_allowed = torch.isin(terrain.terrain_types[env_ids], allowed_ids)
    need_resample = env_ids[~current_allowed]
    if len(need_resample) > 0:
        weights = torch.tensor(
            [generator.sub_terrains[name].proportion for name in allowed_names],
            dtype=torch.float,
            device=env.device,
        )
        weights = weights / torch.clamp(weights.sum(), min=1.0e-6)
        sampled = allowed_ids[torch.multinomial(weights, len(need_resample), replacement=True)]
        terrain.terrain_types[need_resample] = sampled
        terrain.terrain_levels[need_resample] = 0
        terrain.env_origins[need_resample] = terrain.terrain_origins[
            terrain.terrain_levels[need_resample], terrain.terrain_types[need_resample]
        ]
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "level_cap": torch.tensor(float(max_level), device=env.device),
        "mean_type_level_cap": torch.mean(terrain._type_level_caps.float()),
        "episode_success_rate": torch.sum(terrain._success_count) / torch.clamp(
            torch.sum(terrain._episode_counts), min=1.0
        ),
        "episode_success_rate_batch": torch.mean(move_up.float()),
        "course_stage": torch.tensor(float(stage), device=env.device),
        "base_ready": torch.tensor(float(base_ready), device=env.device),
        "base_success_rate": base_rate,
        "standard_success_rate": standard_rate,
        "gap_success_rate": gap_rate,
        "plum_blossom_success_rate": plum_rate,
        "high_wall_success_rate": wall_rate,
        "base_success_rate_ema": aggregate_ema(base_terrain_names),
        "standard_success_rate_ema": aggregate_ema(standard_obstacle_names),
        "gap_success_rate_ema": aggregate_ema(("flat_gap",)),
        "plum_blossom_success_rate_ema": aggregate_ema(("rough_plum_blossom",)),
        "high_wall_success_rate_ema": aggregate_ema(("flat_high_wall",)),
        "command_max_x": torch.tensor(command_ranges[stage][0][1], device=env.device),
        "command_max_y": torch.tensor(command_ranges[stage][1][1], device=env.device),
        "command_max_yaw": torch.tensor(command_ranges[stage][2][1], device=env.device),
    }
    for obstacle_name in ("flat_gap", "rough_plum_blossom", "flat_high_wall"):
        if obstacle_name in sub_names:
            obstacle_mask = previous_type_ids == sub_names.index(obstacle_name)
            if obstacle_mask.any():
                result[f"{obstacle_name}_success_rate_batch"] = torch.mean(move_up[obstacle_mask].float())
            obstacle_rate, _ = aggregate((obstacle_name,))
            result[f"{obstacle_name}_success_rate"] = obstacle_rate
    plum_completed = getattr(env, "_plum_blocks_completed", None)
    if plum_completed is not None and "rough_plum_blossom" in sub_names:
        plum_mask = previous_type_ids == sub_names.index("rough_plum_blossom")
        if plum_mask.any():
            result["rough_plum_blossom_blocks_completed"] = torch.mean(plum_completed[env_ids][plum_mask].float())
    for index, name in enumerate(sub_names):
        mask = terrain.terrain_types == index
        if mask.any():
            result[name] = torch.mean(levels[mask])
        result[f"{name}_level_cap"] = terrain._type_level_caps[index].float()
    return result
