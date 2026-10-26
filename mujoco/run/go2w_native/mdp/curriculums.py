from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from minidog_native.config.obstacle_profiles import (
    BASE_TERRAIN_NAMES,
    GATE_GROUPS,
    OBSTACLE_FAMILIES,
    STANDARD_OBSTACLE_NAMES,
    selected_families,
)


# ETH-style terrain curricula use a local success signal to move an episode up
# one row and a failure signal to move it down one row.  The additional cap
# below is per terrain type: a wall can remain at its low row while the flat
# and stair policies continue improving.
EPISODES_PER_DIFFICULTY_UPGRADE = 256
DIFFICULTY_UPGRADE_SUCCESS = 0.75
DIFFICULTY_DOWNGRADE_SUCCESS = 0.45
# Obstacle families promote earlier than the base terrains: their success rate
# starts from zero, so the base threshold of 0.75 would never be reached.
OBSTACLE_UPGRADE_SUCCESS = 0.55
# The stepping-stone family gets a slightly lower bar still.  Its success
# criterion is a full 2.7 m crossing, and every extra row makes that harder, so
# 0.55 left the cap at 1 for 1500 iterations (plum_fixed1: cap 1.0 throughout,
# success 0.00).  With the 7-row ladder the low rows are now rollable, but the
# family should not need 55% *full crossings* to see the next row.
PLUM_UPGRADE_SUCCESS = 0.45
# A stepping-stone row is "cleared" when the policy visits this many of its six
# cubes on average; that alone unlocks the next difficulty row.
PLUM_UPGRADE_BLOCKS = 2.5
# Highest plum row the curriculum may promote to.  The shaped ladder puts the
# "roll -> lift" transition on rows 3-4; row 6 is the competition course
# (0.200 m cubes, 0.405 m wheel-line slot = 2.35x the 0.172 m wheel, 20% static
# support).  The ceiling is raised one row at a time, only after the row below is
# actually solved on the *real* terrain, because plum_pilot3 showed what happens
# otherwise: cap jumped to 6 by iteration 90, ~90% of the samples went to rows whose
# success rate was exactly 0.000, and the row it had just learned regressed from
# 1.00 to 0.47.
#   row 4 gate (met by plum_soft2):  s_l4 = 0.909 (last 100 iterations)
#   row 5 gate (met by plum_L5):     s_l5 = 0.915 (iterations 1299-1498),
#                                    progress_l5 1.230, blocks_l5 5.92/6
#   -> ceiling 5 -> 6, the competition course.
PLUM_MAX_LEVEL = 6
OBSTACLE_TERRAIN_NAMES = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "random_grid",
    "sloped_terrain",
    "flat_gap",
    "rough_plum_blossom",
    "flat_high_wall",
)

# The mixed task uses a bounded learning-progress sampler inside the normal
# terrain release gates.  This keeps the run single-stage while still giving
# more rollouts to terrain/level cells whose score is changing.  The sampler
# state lives on the terrain entity and is persisted by ``CsvMetricsRunner``.
LP_EPISODES_PER_UPDATE = 512
LP_EMA_ALPHA = 0.20
LP_EPSILON = 0.15
LP_TEMPERATURE_FLOOR = 0.01
# The stepping-stone family used to have an obstacle-free level 0 that the
# sampler escaped to (98% of plum episodes in plum_step3, blocks completed fell to
# 0.1 while the reported success rate rose to 0.34).  That hole is now closed in
# the terrain itself: every course row holds the course geometry (row 0 is a
# continuous 0.60 m slab, the first real gap is row 3).  The floor is therefore
# disabled; it only ever compensated for the missing course.
COURSE_LEVEL_FLOOR = 0.0
LP_MAX_PROBABILITY = 0.08
LP_UPDATE_WEIGHT = 0.35
LP_HISTORY_REVISIT = 0.15

# While a single-obstacle run is focused (its obstacle is already released),
# only this share of episodes still goes to the base terrains; the rest goes to
# the obstacle.  Without it the sampler pool is 3 base terrains + 1 obstacle,
# i.e. ~75% easy locomotion, and the policy can farm locomotion rewards instead
# of learning the obstacle it is supposed to learn.
FOCUS_BASE_SHARE = 0.2

# ``GO2W_PLUM_PILOT=1`` skips the basic-locomotion gate and trains the plum
# column from iteration 0.  The gate is a measured 2.5 m forward-distance test on
# flat/rough ground, and in plum_bootcamp2 it opened at iteration 212 and then
# collapsed (base EMA 0.78 -> 0.21), so ~90% of the 300 iterations went to base
# locomotion and the course was sampled only after iteration 212.  The flag is
# off by default: normal runs keep the gate.
PLUM_PILOT_ENV_VAR = "GO2W_PLUM_PILOT"
# Evidence window for promoting a plum row in pilot mode.  With ~98% of the
# batch on the course, 256 episodes is under two iterations -- far too little to
# justify a harder row -- so the pilot waits for this many instead.
PLUM_PILOT_EPISODES_PER_UPGRADE = 1024

_pilot_warned = False


def plum_pilot() -> bool:
    global _pilot_warned
    enabled = os.environ.get(PLUM_PILOT_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enabled and not _pilot_warned:
        _pilot_warned = True
        families = selected_families()
        if len(families) != 1:
            # The gate is skipped either way, but focus mode needs exactly one
            # family; without it the pilot still spends most rollouts on the
            # base terrains or on other obstacles.
            print(
                f"[Go2W] {PLUM_PILOT_ENV_VAR}=1 expects a single-family run "
                f"(MINIDOG_OBSTACLES=plum); current profile is "
                f"{'+'.join(families) or 'none'}.",
                flush=True,
            )
    return enabled

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _ensure_learning_progress_state(terrain, num_types: int, num_levels: int, device: torch.device) -> None:
    """Create the LP-ACRL task table without coupling it to the runner."""
    task_count = num_types * num_levels
    probabilities = getattr(terrain, "_lp_probabilities", None)
    if probabilities is not None and probabilities.shape == (task_count,):
        return
    terrain._lp_probabilities = torch.full(
        (task_count,), 1.0 / task_count, dtype=torch.float, device=device
    )
    terrain._lp_reward_ema = torch.zeros(task_count, dtype=torch.float, device=device)
    terrain._lp_progress = torch.zeros(task_count, dtype=torch.float, device=device)
    terrain._lp_has_estimate = torch.zeros(task_count, dtype=torch.bool, device=device)
    terrain._lp_stage_sum = torch.zeros(task_count, dtype=torch.float, device=device)
    terrain._lp_stage_count = torch.zeros(task_count, dtype=torch.long, device=device)
    terrain._lp_stage_episodes = 0
    terrain._lp_stage = 0


def _finish_learning_progress_stage(terrain) -> None:
    """Update task probabilities after a bounded number of completed episodes."""
    observed = terrain._lp_stage_count >= 2
    if not bool(torch.any(observed)):
        terrain._lp_stage_sum.zero_()
        terrain._lp_stage_count.zero_()
        terrain._lp_stage_episodes = 0
        return

    means = terrain._lp_stage_sum / terrain._lp_stage_count.clamp_min(1)
    first = observed & ~terrain._lp_has_estimate
    repeated = observed & terrain._lp_has_estimate
    terrain._lp_progress.zero_()
    terrain._lp_reward_ema[first] = means[first]
    if bool(torch.any(repeated)):
        old = terrain._lp_reward_ema[repeated].clone()
        updated = (1.0 - LP_EMA_ALPHA) * old + LP_EMA_ALPHA * means[repeated]
        terrain._lp_progress[repeated] = updated - old
        terrain._lp_reward_ema[repeated] = updated
    terrain._lp_has_estimate[observed] = True

    active = terrain._lp_progress != 0.0
    if bool(torch.any(active)):
        scale = torch.quantile(terrain._lp_progress[active].abs(), 0.75)
    else:
        scale = torch.zeros((), device=terrain._lp_probabilities.device)
    temperature = torch.clamp(scale, min=LP_TEMPERATURE_FLOOR)
    adaptive = torch.softmax(terrain._lp_progress / temperature, dim=0)
    uniform = torch.full_like(adaptive, 1.0 / adaptive.numel())
    target = (1.0 - LP_EPSILON) * adaptive + LP_EPSILON * uniform
    target = torch.clamp(target, min=1.0e-8)
    # Redistribute capped probability mass instead of renormalizing a clipped
    # vector, which could silently push the largest task above the cap again.
    target = target / target.sum().clamp_min(1.0e-8)
    for _ in range(8):
        excess = torch.clamp(target - LP_MAX_PROBABILITY, min=0.0).sum()
        target = torch.clamp(target, max=LP_MAX_PROBABILITY)
        free = target < (LP_MAX_PROBABILITY - 1.0e-7)
        if not bool(free.any()) or float(excess) <= 1.0e-8:
            break
        target = target + excess * free.float() / free.float().sum()
    target = target / target.sum().clamp_min(1.0e-8)
    terrain._lp_probabilities.mul_(1.0 - LP_UPDATE_WEIGHT).add_(
        target, alpha=LP_UPDATE_WEIGHT
    )
    terrain._lp_probabilities.div_(terrain._lp_probabilities.sum().clamp_min(1.0e-8))
    terrain._lp_stage_sum.zero_()
    terrain._lp_stage_count.zero_()
    terrain._lp_stage_episodes = 0
    terrain._lp_stage += 1


def _submit_learning_progress(
    terrain,
    task_ids: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    """Accumulate episode scores and periodically update LP-ACRL weights."""
    valid = (task_ids >= 0) & torch.isfinite(scores)
    if not bool(torch.any(valid)):
        return
    ids = task_ids[valid]
    values = scores[valid]
    offset = 0
    while offset < len(ids):
        take = min(LP_EPISODES_PER_UPDATE - terrain._lp_stage_episodes, len(ids) - offset)
        segment = slice(offset, offset + take)
        segment_ids = ids[segment]
        terrain._lp_stage_sum.scatter_add_(0, segment_ids, values[segment])
        terrain._lp_stage_count.scatter_add_(
            0, segment_ids, torch.ones_like(segment_ids, dtype=torch.long)
        )
        terrain._lp_stage_episodes += int(take)
        offset += take
        if terrain._lp_stage_episodes == LP_EPISODES_PER_UPDATE:
            _finish_learning_progress_stage(terrain)


def _sample_learning_progress_tasks(
    terrain,
    env_ids: torch.Tensor,
    allowed_ids: torch.Tensor,
    max_level: int,
    level_caps: torch.Tensor | None = None,
    revisit_ids: torch.Tensor | None = None,
    focus_ids: torch.Tensor | None = None,
    focus_base_share: float = 1.0,
) -> None:
    """Assign terrain family and level jointly from the LP task table.

    ``focus_ids`` (with ``focus_base_share`` < 1) down-weights every task that
    does not belong to the focused obstacle family, so a single-obstacle run
    spends its rollouts on that obstacle instead of on easy locomotion.
    """
    if len(env_ids) == 0 or len(allowed_ids) == 0:
        return
    num_levels = int(terrain.terrain_origins.shape[0])
    task_mask = torch.zeros_like(terrain._lp_probabilities, dtype=torch.bool)
    for terrain_id in allowed_ids.tolist():
        start = int(terrain_id) * num_levels
        upper_level = min(int(max_level), num_levels - 1)
        if level_caps is not None:
            # Per-family caps are stricter than the release-stage cap.  The
            # sampler must not jump a newly released family to an unmastered row.
            cap = int(torch.clamp(level_caps[terrain_id], 0, num_levels - 1).item())
            upper_level = min(upper_level, cap)
        task_mask[start : start + upper_level + 1] = True
    weights = terrain._lp_probabilities * task_mask.float()
    if focus_ids is not None and len(focus_ids) > 0:
        focus_mask = torch.zeros_like(task_mask)
        for terrain_id in focus_ids.tolist():
            start = int(terrain_id) * num_levels
            upper_level = min(int(max_level), num_levels - 1)
            if level_caps is not None:
                cap = int(torch.clamp(level_caps[terrain_id], 0, num_levels - 1).item())
                upper_level = min(upper_level, cap)
            focus_mask[start : start + upper_level + 1] = task_mask[start : start + upper_level + 1]
        focus_sum = float(weights[focus_mask].sum()) if bool(focus_mask.any()) else 0.0
        base_sum = float(weights[~focus_mask].sum())
        if focus_sum > 1.0e-8 and base_sum > 1.0e-8:
            # Rescale the two groups so the non-focus share is exactly
            # ``focus_base_share`` of all sampling mass, keeping the relative
            # shape inside each group.  Clamped below 1.0 so the scale factor
            # cannot divide by zero, and above 0 so base terrain never
            # disappears completely.
            share = min(max(float(focus_base_share), 0.02), 0.95)
            scale = (share / (1.0 - share)) * (focus_sum / base_sum)
            weights = torch.where(
                focus_mask,
                weights,
                weights * torch.full_like(weights, scale),
            )
    if float(weights.sum()) <= 1.0e-8:
        weights = task_mask.float()
    # Keep a floor on rows that actually contain the obstacle (see
    # COURSE_LEVEL_FLOOR).  Mass is moved from the obstacle-free level 0 rather
    # than added, so the relative shape above level 0 is preserved.
    for terrain_id in allowed_ids.tolist():
        start = int(terrain_id) * num_levels
        upper_level = min(int(max_level), num_levels - 1)
        if level_caps is not None:
            cap = int(torch.clamp(level_caps[terrain_id], 0, num_levels - 1).item())
            upper_level = min(upper_level, cap)
        if upper_level < 1:
            continue
        family = weights[start : start + upper_level + 1]
        total = float(family.sum())
        if total <= 1.0e-8:
            continue
        course = float(family[1:].sum())
        target = COURSE_LEVEL_FLOOR * total
        if course >= target:
            continue
        if course > 1.0e-8:
            weights[start + 1 : start + upper_level + 1] = family[1:] * (target / course)
        else:
            weights[start + 1 : start + upper_level + 1] = target / float(upper_level)
        weights[start] = max(total - target, 0.0)
    if float(weights.sum()) <= 1.0e-8:
        weights = task_mask.float()
    weights = weights / weights.sum().clamp_min(1.0e-8)
    task_ids = torch.multinomial(weights, len(env_ids), replacement=True)
    # Deterministic course floor: rewriting the weights is not enough, because the
    # history-replay override below replaces part of the batch and because a
    # collapsed LP distribution can still leave almost all mass on level 0.  Here
    # the already-drawn batch is corrected directly, so the guarantee holds no
    # matter what the sampler proposed.
    for terrain_id in allowed_ids.tolist():
        start = int(terrain_id) * num_levels
        upper_level = min(int(max_level), num_levels - 1)
        if level_caps is not None:
            cap = int(torch.clamp(level_caps[terrain_id], 0, num_levels - 1).item())
            upper_level = min(upper_level, cap)
        if upper_level < 1:
            continue
        in_family = (task_ids >= start) & (task_ids <= start + upper_level)
        family_total = int(in_family.sum())
        if family_total == 0:
            continue
        at_level0 = in_family & (task_ids == start)
        on_course = family_total - int(at_level0.sum())
        required = int(math.ceil(COURSE_LEVEL_FLOOR * family_total))
        missing = required - on_course
        if missing <= 0:
            continue
        candidates = torch.nonzero(at_level0, as_tuple=False).flatten()[:missing]
        if candidates.numel() > 0:
            picks = torch.randint(
                1, upper_level + 1, (candidates.numel(),), device=env_ids.device
            )
            task_ids[candidates] = start + picks
    if revisit_ids is not None and len(revisit_ids) > 0 and len(env_ids) > 1:
        revisit_mask = torch.zeros_like(task_mask)
        for terrain_id in revisit_ids.tolist():
            start = int(terrain_id) * num_levels
            upper_level = min(int(max_level), num_levels - 1)
            if level_caps is not None:
                cap = int(torch.clamp(level_caps[terrain_id], 0, num_levels - 1).item())
                upper_level = min(upper_level, cap)
            revisit_mask[start : start + upper_level + 1] = task_mask[
                start : start + upper_level + 1
            ]
        revisit_weights = terrain._lp_probabilities * revisit_mask.float()
        if float(revisit_weights.sum()) > 1.0e-8:
            revisit_weights = revisit_weights / revisit_weights.sum()
            revisit_count = max(1, int(round(LP_HISTORY_REVISIT * len(env_ids))))
            indices = torch.randperm(len(env_ids), device=env_ids.device)[:revisit_count]
            task_ids[indices] = torch.multinomial(
                revisit_weights, len(indices), replacement=True
            )
    terrain.terrain_types[env_ids] = task_ids // num_levels
    terrain.terrain_levels[env_ids] = task_ids.remainder(num_levels)
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]


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

    command_distance = torch.abs(command[env_ids, 0]) * env.max_episode_length_s
    move_down = (distance < fail_distance_ratio * command_distance) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
    }

    terrain_types = getattr(terrain, "terrain_types", None)
    origins = getattr(terrain, "terrain_origins", None)
    if terrain_types is not None and origins is not None:
        sub_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        if origins.shape[1] == len(sub_names):
            for index, name in enumerate(sub_names):
                mask = terrain_types == index
                if mask.any():
                    result[name] = torch.mean(levels[mask])
    return result


def _valid_course_exit(env: ManagerBasedRlEnv, env_ids: torch.Tensor, move_up: torch.Tensor) -> torch.Tensor:
    """Filter curriculum progress to upright, non-failed episode exits.

    ``ManagerBasedRlEnv._reset_idx`` runs ``curriculum_manager.compute`` *before*
    ``termination_manager.reset``, so the termination buffers still describe the
    episode that just ended.  Every episode ends either by timeout (survived the
    full length, the normal successful ending) or by a failure term.  Rejecting
    timeouts as well therefore rejected every possible ending, so no success was
    ever counted and the course could not leave stage 0 in the unified run.
    Only genuine failure terminations disqualify an episode now.
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
    # Only the locomotion terrains are mandatory.  Obstacle families may be
    # switched off through MINIDOG_OBSTACLES, and every obstacle-specific block
    # below is guarded by a name check, so requiring all three special terrains
    # here would silently fall back to the simple curriculum for every modular
    # run and skip the release gates entirely.
    required_names = tuple(name for name in base_terrain_names if name in sub_names)
    if not required_names:
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
    previous_level_ids = terrain.terrain_levels[env_ids].clone()
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
    # Diagnostics for the plum column.  A zero plum success rate on its own is
    # indistinguishable from a broken success channel, so log the ingredients:
    # how far an episode actually got, whether it reached the exit distance, and
    # whether a collision had disqualified it.  These keys are only present on
    # iterations that actually ended a plum episode.
    plum_diag: dict[str, float] = {}
    if len(previous_names) > 0:
        plum_mask = previous_type_ids == previous_names[0]
        # ``_plum_course_success`` is the *instantaneous* exit condition: it is
        # recomputed every step and is therefore already False again by the time an
        # episode resets (the robot crosses the exit, keeps driving, and drops off
        # the far platform).  Reading it here made ``plum_success_frac`` read 0.000
        # on every row even though the crossing reward (+40) paid and
        # ``plum_progress_l0`` was 1.28 -- the "reached but never successful"
        # signature of plum_pilot1.  ``_obstacle_course_success`` is the latched
        # ``course_completion.completed`` flag, exported before RewardManager.reset,
        # so it holds "this episode crossed the exit at least once".
        plum_success = getattr(env, "_obstacle_course_success", None)
        if plum_success is None:
            plum_success = getattr(env, "_plum_course_success", None)
        if plum_success is not None:
            move_up = torch.where(plum_mask, plum_success[env_ids], move_up)
        if bool(plum_mask.any()):
            disp_plum = displacement[plum_mask, 0]
            target_plum = target_distance[plum_mask]
            collision = getattr(env, "_course_collision_seen", None)
            collide_plum = (
                collision[env_ids][plum_mask].float()
                if isinstance(collision, torch.Tensor)
                else torch.zeros_like(disp_plum)
            )
            succ_plum = (
                plum_success[env_ids][plum_mask].float()
                if isinstance(plum_success, torch.Tensor)
                else torch.zeros_like(disp_plum)
            )
            plum_diag = {
                "plum_progress": float((disp_plum / target_plum.clamp_min(1.0e-6)).mean()),
                "plum_reached_frac": float((disp_plum > target_plum).float().mean()),
                "plum_collision_frac": float(collide_plum.mean()),
                "plum_success_frac": float(succ_plum.mean()),
            }
            # Which exit condition blocks a crossing that already reached the
            # exit?  These two separate "tipped over" from "collapsed", and they
            # are the ones the policy can fail while standing on the far platform.
            gravity = asset.data.projected_gravity_b[env_ids][plum_mask]
            tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
            rel_z = (
                asset.data.root_link_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
            )[plum_mask]
            plum_diag["plum_tilt_frac"] = float((tilt < 0.70).float().mean())
            plum_diag["plum_height_frac"] = float((rel_z > 0.14).float().mean())
            # Per-row breakdown.  A family-average success rate is what hid the
            # 1500-iteration deadlock: 0.51 looked like "mid-difficulty progress"
            # while the cap was pinned at row 1.  With one key per row the next
            # run shows immediately which row is stuck, and ``plum_gap_seen_frac``
            # shows whether any environment is on a row with a real gap (row 3+
            # on the 7-row ladder).
            levels_plum = previous_level_ids[plum_mask]
            block_counts = getattr(env, "_plum_blocks_completed", None)
            block_plum = (
                block_counts[env_ids][plum_mask]
                if isinstance(block_counts, torch.Tensor)
                else None
            )
            for level in range(int(terrain.terrain_origins.shape[0])):
                level_mask = levels_plum == level
                if not bool(level_mask.any()):
                    continue
                plum_diag[f"plum_success_frac_l{level}"] = float(
                    succ_plum[level_mask].mean()
                )
                plum_diag[f"plum_progress_l{level}"] = float(
                    (disp_plum[level_mask] / target_plum[level_mask].clamp_min(1.0e-6)).mean()
                )
                if block_plum is not None:
                    plum_diag[f"plum_blocks_l{level}"] = float(block_plum[level_mask].mean())
            plum_diag["plum_gap_seen_frac"] = float((levels_plum >= 3).float().mean())

    if not hasattr(terrain, "_adaptive_initialized"):
        n_types = len(sub_names)
        terrain._adaptive_initialized = False
        terrain._success_ema = torch.zeros(n_types, device=env.device)
        terrain._success_count = torch.zeros(n_types, device=env.device)
        terrain._episode_counts = torch.zeros(n_types, device=env.device)
        terrain._type_level_caps = torch.ones(n_types, dtype=torch.long, device=env.device)
        terrain._type_cap_windows = torch.zeros(n_types, dtype=torch.long, device=env.device)
        terrain._course_stage = 0

    _ensure_learning_progress_state(
        terrain,
        len(sub_names),
        int(terrain.terrain_origins.shape[0]),
        env.device,
    )

    # The first reset has no completed episode to score. Do not contaminate
    # the statistics with the random terrain assignment made at construction.
    had_completed_episode = bool(terrain._adaptive_initialized)
    if had_completed_episode:
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

    # Attribute a bounded episode score to the exact terrain/level cell that
    # produced it.  The dense progress component keeps LP-ACRL informative
    # before binary completion becomes common on gaps and walls.
    if had_completed_episode and len(env_ids) > 0:
        progress_score = torch.clamp(distance / target_distance.clamp_min(1.0e-6), 0.0, 1.0)
        episode_score = 0.70 * move_up.float() + 0.30 * progress_score
        num_levels = int(terrain.terrain_origins.shape[0])
        task_ids = previous_type_ids * num_levels + previous_level_ids.clamp(0, num_levels - 1)
        _submit_learning_progress(terrain, task_ids, episode_score)

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

    max_generated_level = terrain.terrain_origins.shape[0] - 1
    plum_index = sub_names.index("rough_plum_blossom") if "rough_plum_blossom" in sub_names else -1
    # Difficulty caps follow the recent success rate (EMA).  A lifetime average
    # can never recover from an early failure phase, which would pin every
    # family to its lowest row even after the policy has improved.
    type_rates = terrain._success_ema
    window_sizes = torch.full_like(
        terrain._episode_counts, float(EPISODES_PER_DIFFICULTY_UPGRADE)
    )
    if plum_index >= 0 and plum_pilot():
        window_sizes[plum_index] = float(PLUM_PILOT_EPISODES_PER_UPGRADE)
    completed_windows = torch.floor(
        terrain._episode_counts / window_sizes
    ).to(torch.long)
    new_windows = completed_windows > terrain._type_cap_windows
    # Obstacle families get their own, lower promotion bar.  With the flat 0.75
    # threshold the plum cap sat at 1 for 2000 iterations because its success
    # rate was pinned at zero, so the policy never saw a harder course row and
    # could never learn one -- a deadlock, not a difficulty setting.
    rate_thresholds = torch.full_like(type_rates, DIFFICULTY_UPGRADE_SUCCESS)
    for index, name in enumerate(sub_names):
        if name == "rough_plum_blossom":
            rate_thresholds[index] = PLUM_UPGRADE_SUCCESS
        elif name in OBSTACLE_TERRAIN_NAMES:
            rate_thresholds[index] = OBSTACLE_UPGRADE_SUCCESS
    can_upgrade = new_windows & (type_rates >= rate_thresholds)
    # Progress-based promotion for the stepping-stone block.  "Cleared a row of
    # cubes" is by itself sufficient evidence to unlock a harder row, and it must
    # not wait for the success channel: in an 800-iteration run the policy cleared
    # all six cubes on the easiest row at iteration 328, yet the success rate
    # stayed 0 (the exit criterion was still rejecting it), so the cap stayed at 1
    # and rows 2-5 were never sampled.  Every row now contains a course
    # (``course_start_difficulty = 0``), so level 0 counts too: excluding it
    # silently disabled this promotion on the row every environment starts on.
    plum_block_rate = 0.0
    plum_completed_now = getattr(env, "_plum_blocks_completed", None)
    if plum_index >= 0 and isinstance(plum_completed_now, torch.Tensor) and len(env_ids) > 0:
        course_rows = (previous_type_ids == plum_index) & (previous_level_ids >= 0)
        if bool(course_rows.any()):
            plum_block_rate = float(plum_completed_now[env_ids][course_rows].mean())
            if plum_block_rate >= PLUM_UPGRADE_BLOCKS:
                plum_indicator = torch.zeros_like(new_windows)
                plum_indicator[plum_index] = True
                can_upgrade = can_upgrade | (new_windows & plum_indicator)
    obstacle_types = torch.tensor(
        [name in OBSTACLE_TERRAIN_NAMES for name in sub_names],
        dtype=torch.bool,
        device=env.device,
    )
    # Obstacle caps are ratchets.  Demoting them here fought the sampler and
    # re-created the deadlock: a family just promoted to row 3 fails a few times,
    # is pushed back to row 2, and can never accumulate evidence on the row it is
    # supposed to learn.  Base terrains keep the normal up/down curriculum.
    should_downgrade = (
        new_windows
        & (completed_windows >= 2)
        & (type_rates < DIFFICULTY_DOWNGRADE_SUCCESS)
        & ~obstacle_types
    )
    caps = terrain._type_level_caps
    # Per-family ceiling.  The plum ladder is shaped so that rows 0-3 are the
    # "roll / first lift" region and rows 4-6 are the competition-size course.  A
    # 1000-iteration pilot climbed to row 6 by iteration 90 and then spent ~90% of
    # its samples on rows 4-6, whose success rate was exactly 0.000, while the row
    # it had just learned (row 2) regressed from 1.00 back to 0.47.  Holding the
    # plum column at row 4 keeps sampling on "learned (2) / being learned (3-4)".
    # Other families keep the full range.
    type_ceilings = torch.full_like(caps, max_generated_level)
    if plum_index >= 0:
        type_ceilings[plum_index] = min(PLUM_MAX_LEVEL, max_generated_level)
    caps = torch.where(can_upgrade, torch.clamp(caps + 1, max=type_ceilings), caps)
    caps = torch.where(should_downgrade, torch.clamp(caps - 1, min=1), caps)
    terrain._type_level_caps = caps
    terrain._type_cap_windows = torch.maximum(terrain._type_cap_windows, completed_windows)

    # Gates are measured on the *recent* success rate (EMA over roughly the last
    # 600-900 episodes).  The lifetime average was used before, which meant one
    # bad opening phase could keep the course at stage 0 forever.  The episode
    # count still guards against releasing an obstacle from a lucky few rollouts.
    base_ema = aggregate_ema(base_terrain_names)
    pilot = plum_pilot()
    if pilot:
        # Start on the course.  The gate is still computed and logged so the
        # base success rate stays visible, but it cannot delay the plum column.
        base_ready = True
    else:
        base_ready = bool(base_count.item() >= 2048 and base_ema.item() >= 0.75)

    # Curriculum releases are ratchets: once a gate opens it stays open for the
    # rest of the run.  The gate rate is a per-reset-batch EMA, so a handful of
    # failures on a newly promoted base row can push it back below the threshold.
    # Measured on an 800-iteration run: the stage flapped open/closed every few
    # dozen iterations, so the plum course was only sampled ~30% of the time and
    # the policy drifted back to optimising flat-ground locomotion (episode
    # reward rose 72 -> 95 while blocks completed fell 2.16 -> 0.04).
    released = getattr(terrain, "_released_gates", None)
    if not isinstance(released, dict):
        released = {}
        terrain._released_gates = released

    def ratchet(key: str, ready: bool) -> bool:
        if ready:
            released[key] = True
        return bool(released.get(key, False)) or bool(ready)

    base_ready = ratchet("base", base_ready)

    def gate_ready(families: tuple[str, ...], min_episodes: int, threshold: float) -> bool:
        names = tuple(
            name
            for family in families
            for name in OBSTACLE_FAMILIES[family]
            if name in sub_names
        )
        if not names:
            # Family not used by this run: it must not block any later stage.
            return True
        _, count = aggregate(names)
        return bool(count.item() >= float(min_episodes) and aggregate_ema(names).item() >= threshold)

    # The release table is assembled from the obstacle families enabled in this
    # run.  Disabling a family removes its stage entirely, so a plum-only run
    # unlocks plum blossom right after the base locomotion gate.
    stage_profiles: list[tuple[tuple[str, ...], int]] = [(tuple(base_terrain_names), 1)]
    gates: list[bool] = [base_ready]
    allowed = list(base_terrain_names)
    for _gate_name, families, min_episodes, threshold, unlock_level in GATE_GROUPS:
        names = tuple(
            name
            for family in families
            for name in OBSTACLE_FAMILIES[family]
            if name in sub_names
        )
        if not names:
            continue
        gates.append(ratchet(_gate_name, gate_ready(families, min_episodes, threshold)))
        allowed.extend(names)
        stage_profiles.append((tuple(dict.fromkeys(allowed)), unlock_level))
    stage_profiles.append((tuple(dict.fromkeys(allowed)), max_generated_level))

    stage = next((index for index, ready in enumerate(gates) if not ready), len(gates))
    stage = min(stage, len(stage_profiles) - 1)
    allowed_names, max_level = stage_profiles[stage]

    max_level = min(max_level, terrain.terrain_origins.shape[0] - 1)
    allowed_names = tuple(dict.fromkeys(name for name in allowed_names if name in sub_names))
    allowed_ids = torch.tensor([sub_names.index(name) for name in allowed_names], dtype=torch.long, device=env.device)
    terrain._course_stage = stage

    # The base gate is a forward-distance test (2.5 m), so the stage-0 speed
    # ceiling sets how long an episode must survive to pass it.  At the old
    # 0.22 m/s ceiling that was >=11.4 s, while plum_bootcamp1 episodes lasted
    # 6.6 s and never scored.  Base locomotion therefore trains at 0.30-0.60
    # m/s.  The plum row only raises forward speed: the course is driven
    # straight, with the 0.284 m wheel track straddling the 0.20 m gutter
    # between the two staggered rows, so lateral and yaw commands stay small.
    # The 0.20 m/s floor exists because the real Go2W has a low-speed dead
    # zone: a policy trained to track 0.10 m/s asks for wheel speeds the
    # hardware never executes and stalls on the first stone.  The 0.35 m/s
    # ceiling still gives 7.7-13.5 s for the 2.7 m course in a 30 s episode.
    # Other obstacle families get a middle row.
    base_command = ((0.30, 0.60), (-0.04, 0.04), (-0.15, 0.15))
    obstacle_command = ((0.15, 0.30), (-0.04, 0.04), (-0.15, 0.15))
    plum_command = ((0.20, 0.35), (-0.02, 0.02), (-0.10, 0.10))
    if stage == 0:
        command_row = base_command
    elif "rough_plum_blossom" in allowed_names:
        command_row = plum_command
    else:
        command_row = obstacle_command
    try:
        command_cfg = env.command_manager.get_term_cfg(command_name)
        command_cfg.ranges.lin_vel_x = command_row[0]
        command_cfg.ranges.lin_vel_y = command_row[1]
        command_cfg.ranges.ang_vel_z = command_row[2]
    except (AttributeError, KeyError):
        # Keep the curriculum usable with environments that do not use a
        # UniformVelocityCommand term.
        pass

    command_distance = torch.abs(command[env_ids, 0]) * env.max_episode_length_s
    move_down = (distance < fail_distance_ratio * command_distance) & ~move_up
    # The sampler above already chose a level for every resetting environment.
    # Demoting on the same reset cancels that choice: failing on the coarse
    # stepping-stone row put the environment straight back on the obstacle-free
    # level 0, which is why the measured mean level stayed at 0.08 while the
    # sampler was asking for 60% course rows.  Obstacle families are therefore
    # only ever promoted here -- exploration is the sampler's job, not this term's.
    on_obstacle = obstacle_types[terrain.terrain_types[env_ids]]
    move_down = move_down & ~on_obstacle
    terrain.terrain_levels[env_ids] += move_up.to(torch.long) - move_down.to(torch.long)
    type_caps = terrain._type_level_caps[terrain.terrain_types[env_ids]]
    terrain.terrain_levels[env_ids] = torch.minimum(
        torch.clamp(terrain.terrain_levels[env_ids], min=0, max=max_level), type_caps
    )

    # Sample the next episode from the released terrain/level task table.
    # Flat and rough terrain remain eligible for history replay after harder
    # families are released; no separate optimizer/checkpoint stage is needed.
    revisit_ids = torch.tensor(
        [sub_names.index(name) for name in base_terrain_names if name in sub_names],
        dtype=torch.long,
        device=env.device,
    )

    # Focus mode: when the run trains exactly one obstacle family and that family
    # is already released, almost every episode must be that obstacle.  Without
    # this the sampler pool is 3 base terrains + 1 obstacle, so ~75% of episodes
    # are easy locomotion and the policy can farm the locomotion rewards instead
    # of learning the obstacle.  The warm-up (stage 0) is untouched: the obstacle
    # is not released yet, so the robot still learns to drive first.
    focus_ids = None
    focused_families = selected_families()
    if len(focused_families) == 1 and stage >= 1:
        focus_names = tuple(
            name
            for family in focused_families
            for name in OBSTACLE_FAMILIES[family]
            if name in sub_names
        )
        if focus_names:
            focus_ids = torch.tensor(
                [sub_names.index(name) for name in focus_names],
                dtype=torch.long,
                device=env.device,
            )

    _sample_learning_progress_tasks(
        terrain,
        env_ids,
        allowed_ids,
        max_level,
        level_caps=terrain._type_level_caps,
        # Forced history replay would re-inflate the base share that focus mode
        # deliberately lowers, so it is disabled while focusing.
        revisit_ids=None if focus_ids is not None else revisit_ids,
        focus_ids=focus_ids,
        # The sampler clamps this at 0.02, so a pilot batch is ~98% course.
        focus_base_share=0.02 if pilot else FOCUS_BASE_SHARE,
    )
    terrain._focus_mode = focus_ids is not None

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
        "focus_mode": torch.tensor(float(focus_ids is not None), device=env.device),
        "plum_pilot": torch.tensor(float(pilot), device=env.device),
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
        "lp_stage": torch.tensor(float(terrain._lp_stage), device=env.device),
        "lp_stage_progress": torch.tensor(
            float(terrain._lp_stage_episodes) / float(LP_EPISODES_PER_UPDATE),
            device=env.device,
        ),
        "lp_effective_tasks": torch.exp(
            -torch.sum(
                terrain._lp_probabilities
                * torch.log(terrain._lp_probabilities.clamp_min(1.0e-8))
            )
        ),
        "lp_max_probability": terrain._lp_probabilities.max(),
        "lp_mean_abs_progress": terrain._lp_progress.abs().mean(),
        "lp_task_coverage": terrain._lp_stage_count.gt(0).float().mean(),
        "lp_history_revisit_probability": torch.tensor(LP_HISTORY_REVISIT, device=env.device),
        "command_max_x": torch.tensor(command_row[0][1], device=env.device),
        "command_max_y": torch.tensor(command_row[1][1], device=env.device),
        "command_max_yaw": torch.tensor(command_row[2][1], device=env.device),
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
    for key, value in plum_diag.items():
        result[key] = torch.tensor(value, device=env.device)
    return result
