from __future__ import annotations

import torch

from minidog_native.mdp.rewards import *  # noqa: F401,F403
# ``import *`` skips underscore-prefixed names, so the shared helpers have to be
# imported explicitly once here instead of inside every reward function.
from minidog_native.mdp.rewards import _contact_flags
from go2w_native.robot_cfg import GO2W_FOOT_BODY_NAMES, WHEEL_OFFSETS
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.lab_api.math import quat_apply_inverse


OBSTACLE_TERRAIN_NAMES = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "stairs_up",
    "stairs_down",
    "random_grid",
    "blocks",
    "sloped_terrain",
    "slope",
    "flat_gap",
    "rough_plum_blossom",
    "flat_high_wall",
)


def wheel_mode_leg_activity_l2(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.03,
    position_weight: float = 0.25,
    velocity_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize unnecessary leg motion outside generated obstacle terrain."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    active = (
        torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    ) > command_threshold
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    cost = position_weight * torch.mean(torch.square(q - q0), dim=1)
    cost = cost + velocity_weight * torch.mean(torch.square(qd), dim=1)

    terrain = getattr(env.scene, "terrain", None)
    obstacle = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain is not None and terrain.terrain_types is not None:
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        for name in OBSTACLE_TERRAIN_NAMES:
            if name in names:
                obstacle |= terrain.terrain_types == names.index(name)
    return cost * active.float() * (~obstacle).float()


def wheel_velocity_tracking(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.10,
    command_threshold: float = 0.03,
    std: float = 1.5,
    obstacle_scale: float = 0.25,
) -> torch.Tensor:
    """Track the commanded forward speed with wheel rotation.

    The absolute value is used because left/right wheel encoder signs depend
    on the wheel mesh/joint axis convention. Direction and yaw are still
    constrained by the base velocity tracking term.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    target = torch.abs(command[:, 0]).unsqueeze(1) / max(wheel_radius, 1.0e-3)
    error = torch.mean(torch.square(torch.abs(wheel_vel) - target), dim=1)
    active = (torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > command_threshold
    terrain = getattr(env.scene, "terrain", None)
    obstacle = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain is not None and terrain.terrain_types is not None:
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        for name in OBSTACLE_TERRAIN_NAMES:
            if name in names:
                obstacle |= terrain.terrain_types == names.index(name)
    obstacle_scale = min(max(float(obstacle_scale), 0.0), 1.0)
    mode_scale = torch.where(obstacle, torch.full_like(error, obstacle_scale), torch.ones_like(error))
    return torch.exp(-error / max(std, 1.0e-3) ** 2) * active.float() * mode_scale


def obstacle_feet_air_time(
    env,
    sensor_name: str,
    threshold_min: float = 0.08,
    threshold_max: float = 0.20,
    command_name: str | None = None,
    command_threshold: float = 0.03,
    obstacle_difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Enable swing reward only on a terrain row that contains an obstacle.

    ``threshold_max`` is deliberately short (0.20 s).  The course must be crossed
    by placing one wheel at a time; a longer flight is a jump, and a jump cannot
    be aimed at a 0.20 m cube.  Rewarding air time beyond a deliberate step would
    pay the policy for hopping instead of stepping.

    Wheels remain the default locomotion mode.  At low rows a generated
    obstacle may still be effectively flat (for example a zero-height stair),
    so the reward is additionally gated by the current normalized terrain
    difficulty.  The contact sensor still supplies the actual air-time term;
    this function only decides when it is relevant.
    """
    reward = velocity_mdp.feet_air_time(
        env,
        sensor_name=sensor_name,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        command_name=command_name,
        command_threshold=command_threshold,
    )
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None or terrain.terrain_origins is None:
        return torch.zeros_like(reward)

    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    is_obstacle = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in OBSTACLE_TERRAIN_NAMES:
        if name in names:
            is_obstacle |= terrain.terrain_types == names.index(name)
    level_count = max(int(terrain.terrain_origins.shape[0]) - 1, 1)
    difficulty = terrain.terrain_levels.float() / float(level_count)
    active = is_obstacle & (difficulty >= obstacle_difficulty_threshold)
    return reward * active.float()


def gap_takeoff_progress(
    env,
    command_name: str,
    # FlatGapTerrainCfg spawns at local x=1.20; reward coordinates are
    # relative to that origin rather than absolute terrain coordinates.
    gap_start_x: float = 1.15,
    approach_width: float = 0.45,
    command_threshold: float = 0.05,
    vertical_speed_scale: float = 0.45,
) -> torch.Tensor:
    """Shape a controlled takeoff immediately before a ditch."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None:
        return torch.zeros(env.num_envs, device=env.device)
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if "flat_gap" not in names:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene["robot"]
    displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    command = env.command_manager.get_command(command_name)
    active = command[:, 0] > command_threshold
    approach = torch.exp(-torch.square(displacement - (gap_start_x - 0.12)) / max(approach_width, 1.0e-3) ** 2)
    upward_speed = torch.clamp(asset.data.root_link_lin_vel_w[:, 2], min=0.0)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return (terrain.terrain_types == names.index("flat_gap")).float() * active.float() * approach * upright * torch.clamp(
        upward_speed / max(vertical_speed_scale, 1.0e-3), 0.0, 1.0
    )


def gap_landing_stability(
    env,
    command_name: str,
    # ``None`` derives the exit edge from the generated gap width at the
    # environment's current difficulty instead of assuming the widest ditch.
    gap_end_x: float | None = None,
    landing_width: float = 0.55,
    command_threshold: float = 0.05,
    vertical_speed_scale: float = 0.45,
    spawn_x: float = 1.20,
) -> torch.Tensor:
    """Reward a quiet, upright landing after the ditch exit."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None:
        return torch.zeros(env.num_envs, device=env.device)
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if "flat_gap" not in names:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene["robot"]
    displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    command = env.command_manager.get_command(command_name)
    active = command[:, 0] > command_threshold
    if gap_end_x is None:
        gap_cfg = terrain.cfg.terrain_generator.sub_terrains.get("flat_gap")
        if gap_cfg is None:
            return torch.zeros(env.num_envs, device=env.device)
        level_count = max(int(terrain.terrain_origins.shape[0]) - 1, 1)
        difficulty = (terrain.terrain_levels.float() / float(level_count)).clamp(0.0, 1.0)
        # Same expression as FlatGapTerrainCfg.function, minus its small jitter.
        width = gap_cfg.gap_width_range[0] + difficulty * (
            gap_cfg.gap_width_range[1] - gap_cfg.gap_width_range[0]
        )
        exit_edge = torch.clamp(
            gap_cfg.gap_start_x + width, max=float(gap_cfg.size[0]) - 0.8
        )
        gap_end = exit_edge - spawn_x
    else:
        gap_end = torch.full_like(displacement, float(gap_end_x))
    landing = torch.sigmoid((displacement - gap_end) / max(landing_width, 1.0e-3))
    vertical_stability = torch.exp(-torch.square(asset.data.root_link_lin_vel_w[:, 2]) / max(vertical_speed_scale, 1.0e-3) ** 2)
    forward = torch.clamp(asset.data.root_link_lin_vel_b[:, 0] / torch.clamp(command[:, 0], min=0.08), 0.0, 1.0)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return (terrain.terrain_types == names.index("flat_gap")).float() * active.float() * landing * vertical_stability * forward * upright


def high_wall_clearance_progress(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    # FlatHighWallTerrainCfg spawns at local x=1.5. These positions are the
    # three wall centers relative to that spawn origin.
    wall_centers_x: tuple[float, ...] = (1.40, 2.95, 4.50),
    wheel_radius: float = 0.086,
    clearance_margin: float = 0.0,
    clearance_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Encourage leg clearance and forward motion at each wall front.

    ``foot_z`` is the wheel axle, so clearing a wall of height ``h`` needs an
    axle height of ``h + wheel_radius``.  The required clearance is therefore
    derived from the wall height at the environment's difficulty (instead of a
    fixed 0.13-0.42 m ramp that a wheel could satisfy without ever clearing the
    wall).  ``clearance_range`` keeps the old fixed behaviour available.
    """
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None:
        return torch.zeros(env.num_envs, device=env.device)
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if "flat_high_wall" not in names:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    command = env.command_manager.get_command(command_name)
    active = command[:, 0] > command_threshold
    nearest = torch.stack([torch.abs(displacement - center) for center in wall_centers_x], dim=1).amin(dim=1)
    near_wall = torch.exp(-torch.square(nearest) / 0.18**2)
    foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, None, 2]
    level_count = max(int(terrain.terrain_origins.shape[0]) - 1, 1)
    difficulty = (terrain.terrain_levels.float() / float(level_count)).clamp(0.0, 1.0)
    wall_present = torch.ones_like(active)
    if clearance_range is None:
        wall_cfg = terrain.cfg.terrain_generator.sub_terrains.get("flat_high_wall")
        if wall_cfg is not None:
            start = float(wall_cfg.wall_start_difficulty)
            # Rows below the wall threshold contain no wall at all; rewarding
            # clearance there would pay for hopping on flat ground.
            wall_present = difficulty >= start
            progress = ((difficulty - start) / max(1.0 - start, 1.0e-6)).clamp(0.0, 1.0)
            height_lo, height_hi = wall_cfg.wall_height_range
            wall_height = height_lo + progress * (height_hi - height_lo)
            required_clearance = wall_height + wheel_radius + clearance_margin
        else:
            required_clearance = 0.08 + 0.42 * difficulty + wheel_radius + clearance_margin
    else:
        required_clearance = clearance_range[0] + difficulty * (
            clearance_range[1] - clearance_range[0]
        )
    clearance = torch.sigmoid((torch.amax(foot_z, dim=1) - required_clearance) / 0.025)
    forward = torch.clamp(asset.data.root_link_lin_vel_b[:, 0] / torch.clamp(command[:, 0], min=0.08), 0.0, 1.0)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return (
        (terrain.terrain_types == names.index("flat_high_wall")).float()
        * active.float()
        * wall_present.float()
        * near_wall
        * clearance
        * forward
        * upright
    )


# --------------------------------------------------------------------------- #
# Obstacle-aware reward balance
#
# On flat ground the dominant reward is "hold the commanded velocity with the
# wheels" (track_lin_vel alone measured 41% of the episode return).  On the
# stepping stones that same incentive is actively harmful: planting a wheel on a
# 0.20 m cube takes ~0.5 s, during which the robot falls behind the command, so
# keeping the wheels spinning and sliding forward pays far more than stepping.
# The helpers below damp the locomotion terms exactly on the obstacle rows and
# add the foothold-shaping terms the reference projects use
# (legged_robot_competition ``feet_air_time``, AME ``feet_too_near`` and
# ``foot_clearance_reward``).
# --------------------------------------------------------------------------- #
def obstacle_row_mask(
    env: ManagerBasedRlEnv,
    difficulty_threshold: float = 0.0,
    terrain_names: tuple[str, ...] = ("rough_plum_blossom",),
) -> torch.Tensor:
    """Environments standing on an obstacle row that actually has the obstacle.

    ``terrain_names`` defaults to the stepping-stone block only.  It must not
    cover stairs, blocks or slopes: damping the locomotion reward there would
    stop the policy from ever learning to roll over them, which is exactly what
    the mixed task needs.
    """
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None:
        return mask
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    for name in terrain_names:
        if name in names:
            mask |= terrain.terrain_types == names.index(name)
    levels = getattr(terrain, "terrain_levels", None)
    if levels is not None:
        num_rows = max(int(terrain.terrain_origins.shape[0]), 2)
        difficulty = levels.float() / float(num_rows - 1)
        mask &= difficulty >= float(difficulty_threshold)
    return mask


def _scale_on_obstacles(env, value: torch.Tensor, obstacle_scale: float, threshold: float) -> torch.Tensor:
    mask = obstacle_row_mask(env, threshold)
    return value * torch.where(
        mask, torch.full_like(value, float(obstacle_scale)), torch.ones_like(value)
    )


def obstacle_scaled_track_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.20,
    obstacle_scale: float = 0.4,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Commanded base-velocity tracking, damped on obstacle rows."""
    return _scale_on_obstacles(
        env, track_linear_velocity(env, std=std, command_name=command_name), obstacle_scale, difficulty_threshold
    )


def obstacle_scaled_forward_motion(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.05,
    min_command: float = 0.08,
    obstacle_scale: float = 0.6,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Forward-motion reward, damped on obstacle rows."""
    value = commanded_forward_motion(
        env, command_name=command_name, command_threshold=command_threshold, min_command=min_command
    )
    return _scale_on_obstacles(env, value, obstacle_scale, difficulty_threshold)


def obstacle_scaled_forward_stall(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.05,
    min_command: float = 0.08,
    obstacle_scale: float = 0.2,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Stall penalty, damped on obstacle rows so stopping to step is affordable."""
    value = commanded_forward_stall(
        env, command_name=command_name, command_threshold=command_threshold, min_command=min_command
    )
    return _scale_on_obstacles(env, value, obstacle_scale, difficulty_threshold)


def obstacle_scaled_flat_orientation(
    env: ManagerBasedRlEnv,
    std: float = 0.5,
    obstacle_scale: float = 0.2,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Keep the base level, but do not fight the roll that the stones require."""
    value = flat_orientation_l2(env)
    return _scale_on_obstacles(env, value, obstacle_scale, difficulty_threshold)


def wheel_lock_on_obstacle(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Penalize wheel spin on obstacle rows.

    On the cubes the wheel has to act as a foot: a spinning wheel slips off a
    0.20 m face instead of planting on it.  ``wheel_velocity_tracking`` is fully
    muted there, and this term actively pushes the hub towards a locked state.
    """
    asset: Entity = env.scene[asset_cfg.name]
    wheel_speed = torch.mean(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return wheel_speed * obstacle_row_mask(env, difficulty_threshold).float()


def wheels_too_near(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    min_separation: float = 0.15,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Penalize two wheels landing inside the same stone gap.

    Ported from AME_Locomotion ``feet_too_near``: with 0.20 m stones and 0.40 m
    gaps a wheel pair that lands within one gap wedges the robot, so the pair
    distances are kept above ``min_separation``.
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2]
    count = pos.shape[1]
    cost = torch.zeros(env.num_envs, device=env.device)
    for i in range(count):
        for j in range(i + 1, count):
            distance = torch.linalg.vector_norm(pos[:, i, :] - pos[:, j, :], dim=1)
            cost = cost + torch.clamp(min_separation - distance, min=0.0)
    return cost * obstacle_row_mask(env, difficulty_threshold).float()


def obstacle_foot_clearance(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    command_threshold: float = 0.05,
    target_height: float = 0.10,
    moving_threshold: float = 0.05,
    max_height: float = 0.30,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Reward lifting a moving swinging wheel above the stone faces.

    Ported from AME_Locomotion ``foot_clearance_reward``: a wheel dragged at
    stone height catches the next cube instead of landing on it.  Gated on an
    active command *and* on the wheel actually moving: without the motion gate the
    term pays for parking one leg in the air, which is what drove plum_step5 into
    a three-wheel stance (mean air time 0.05 -> 0.55 s), constant falls and a value
    loss that blew past 1e5.

    The gate has to be *per wheel*.  Until plum_pilot3 it compared the lifted
    wheel's height against the maximum horizontal speed of all four wheels, so a
    parked wheel paid while the other three drove: mean air time 1.37 s, mean
    episode length 978, +29/iter reward, and the level-2 success rate fell from
    1.00 back to 0.47.  Height and speed now come from the same wheel, and a wheel
    lifted above ``max_height`` (a flail, not a placement over a 0.20 m cube) pays
    nothing.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    active = command[:, 0] > command_threshold
    foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    height = torch.clamp(foot_z - target_height, min=0.0)
    speed = torch.linalg.vector_norm(
        asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
    )
    per_foot = torch.tanh(height / 0.05) * (speed > moving_threshold).float()
    per_foot = per_foot * (height < max_height).float()
    lift = per_foot.amax(dim=1)
    return lift * active.float() * obstacle_row_mask(env, difficulty_threshold).float()


def obstacle_scaled_term(
    env: ManagerBasedRlEnv,
    base_func,
    obstacle_scale: float = 0.2,
    difficulty_threshold: float = 0.0,
    **params,
) -> torch.Tensor:
    """Scale an arbitrary reward term down on the stepping-stone rows.

    ``base_func`` is the term's *original* function, passed through the params so
    this wrapper can never silently substitute a different implementation.  An
    earlier version looked the function up by name in ``minidog_native`` and
    quietly returned zero for ``pose`` (whose real function is mjlab's
    ``variable_posture``), which disabled a +0.8 posture reward and destroyed
    basic locomotion.

    ``pose`` (+0.8), ``leg_mirror`` (-0.25) and ``diagonal_leg_symmetry`` (-0.30)
    are all correct on flat ground and all three oppose the one skill the stones
    need: lift a single wheel and place it elsewhere.
    """
    if isinstance(base_func, type):
        raise TypeError(
            "obstacle_scaled_term only wraps plain reward functions. Class-based "
            "terms (e.g. velocity_mdp.variable_posture) are instantiated by the "
            "reward manager and must not be wrapped."
        )
    value = base_func(env, **params)
    return _scale_on_obstacles(env, value, obstacle_scale, difficulty_threshold)


def obstacle_scaled_upright(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    obstacle_scale: float = 0.25,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """mjlab's ``upright`` reward, damped on the stepping-stone rows.

    ``mjlab.tasks.velocity.mdp.upright`` is a *class* (it is instantiated by the
    reward manager), so ``obstacle_scaled_term`` cannot wrap it.  Holding the base
    perfectly level is correct on flat ground but is the largest remaining
    standing-still income on the stones: with velocity tracking damped there,
    +1.0/step for "do not tilt" outweighs stepping.  This is the same formula as
    the class (world-up projection, no terrain-normal sensor), just as a plain
    function so the obstacle-row scaling applies.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    else:
        body_quat_w = asset.data.root_link_quat_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    value = torch.exp(-xy_squared / std**2)
    return _scale_on_obstacles(env, value, obstacle_scale, difficulty_threshold)


def plum_wheel_on_stone(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Dense credit for every wheel properly resting on a stepping stone.

    At the competition geometry only two wheels can be on stones at any instant
    (the cubes are smaller than the footprint and the rows are staggered by
    0.30 m), so "walking the course" *is* repeated accurate placement.  Paying a
    one-off bonus per newly visited stone values the 25 steps of a careful step
    at almost nothing; paying per step while a wheel is actually on a top face
    makes placement the dominant objective.
    """
    value = getattr(env, "_plum_wheels_on_stone", None)
    if value is None:
        return torch.zeros(env.num_envs, device=env.device)
    return value * obstacle_row_mask(env).float()


def plum_edge_hit(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize wheels that clang into a stone's side face instead of its top."""
    value = getattr(env, "_plum_edge_hits", None)
    if value is None:
        return torch.zeros(env.num_envs, device=env.device)
    return value * obstacle_row_mask(env).float()


def plum_overhang_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """BeamDojo eq. (2): penalize sole sample points that are not over a stone.

    For every wheel in contact, four points around the contact patch are tested
    against the known cube layout.  A wheel that half-overhangs a 0.20 m cube
    still passes the centre-based foothold test but will tip, so this count is
    the signal that keeps the wheel fully supported.
    """
    value = getattr(env, "_plum_overhang", None)
    if value is None:
        return torch.zeros(env.num_envs, device=env.device)
    return value * obstacle_row_mask(env).float()


def plum_next_foothold(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "feet_ground_contact",
    sigma: float = 0.25,
    ahead_margin: float = 0.05,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Reference footstep sequence: guide a swinging wheel towards its next cube.

    The course layout is fixed, so the canonical footstep sequence is known
    analytically -- left wheels land on the upper row cubes (0.6 / 1.2 / 1.8 m from
    the entrance platform) and right wheels on the lower row (0.9 / 1.5 / 2.1 m).
    arXiv 2403.03639 obtains the same information with an MCTS footstep search and
    then trains a policy on the result; a fixed course makes the search redundant.

    Only *swinging* wheels are scored, so the dense contact rewards keep their
    role.  ``sigma = 0.25`` makes the term rise over the last ~0.3 m of the swing,
    which is where the placement decision is made.
    """
    zeros = torch.zeros(env.num_envs, device=env.device)
    mask = obstacle_row_mask(env, difficulty_threshold)
    if not bool(mask.any()):
        return zeros
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or getattr(terrain, "cfg", None) is None:
        return zeros
    course_cfg = terrain.cfg.terrain_generator.sub_terrains.get("rough_plum_blossom")
    centers = getattr(course_cfg, "block_centers", None) if course_cfg is not None else None
    if not centers:
        return zeros

    asset: Entity = env.scene[asset_cfg.name]
    foot_xy = (
        asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2]
        - env.scene.env_origins[:, None, :2]
    )
    contact = _contact_flags(env, sensor_name)[:, : foot_xy.shape[1]] > 0.0

    centers_t = torch.tensor(centers, dtype=torch.float, device=env.device)      # (B, 2)
    dx = centers_t[None, None, :, 0] - foot_xy[:, :, None, 0]                    # (N, F, B)
    dy = centers_t[None, None, :, 1] - foot_xy[:, :, None, 1]
    same_row = (centers_t[None, None, :, 1] > 0.0) == (foot_xy[:, :, None, 1] > 0.0)
    ahead = dx > float(ahead_margin)
    d2 = torch.where(same_row & ahead, dx * dx + dy * dy, torch.full_like(dx, 1.0e6))
    nearest, _ = d2.min(dim=2)                                                   # (N, F)
    approach = torch.exp(-nearest / (float(sigma) * float(sigma)))
    swinging = (~contact).float()
    # Pay for *getting closer*, not for *being close*.  A level reward pays a leg
    # parked in the air above the next cube, and that is exactly the loophole that
    # took plum_step5 down: mean air time rose 0.05 -> 0.55 s, the robot stood on
    # three wheels, fell constantly and the value loss exploded past 1e5.  The
    # potential form below is zero for a hovering leg and unchanged for a real
    # swing, which is the same potential-based shaping argument that keeps the
    # optimum of the task intact.
    potential = (approach * swinging).mean(dim=1)
    previous = getattr(env, "_plum_approach_potential", None)
    step_buf = getattr(env, "episode_length_buf", None)
    previous_step = getattr(env, "_plum_approach_step", None)
    if previous is None or previous.shape != potential.shape:
        env._plum_approach_potential = potential.detach().clone()
        env._plum_approach_step = None if step_buf is None else step_buf.clone()
        return zeros
    # A reset restarts the episode in the middle of a swing, so the stored
    # potential belongs to a different trajectory.  ``episode_length_buf`` drops
    # on reset, which is the cheapest reliable signal available to a plain term.
    valid = torch.ones_like(potential, dtype=torch.bool)
    if step_buf is not None and previous_step is not None and previous_step.shape == step_buf.shape:
        valid = step_buf >= previous_step
    progress = torch.where(valid, potential - previous, torch.zeros_like(potential))
    env._plum_approach_potential = potential.detach().clone()
    if step_buf is not None:
        env._plum_approach_step = step_buf.clone()
    return torch.clamp(progress, min=0.0, max=1.0) * 4.0 * mask.float()


def _build_gait_plan_table(env: ManagerBasedRlEnv, bins: int = 271) -> torch.Tensor | None:
    """Precompute, per course row, which diagonal the reference plan wants.

    This is the cheap form of "MCTS search over contact sequences": for a fixed
    course the statically stable set is a pure function of the body position, so
    the search reduces to enumerating it.  Entry 0 = no static configuration
    (the body must be in a dynamic phase), 1 = FL+RR planted, 2 = FR+RL planted.

    Measured on the competition geometry: the crossing contains five static
    islands of 0.014-0.186 m separated by 0.184-0.200 m of unsupported travel, so
    a quasi-static plan does not exist there -- the table correctly reports 0 over
    those stretches instead of inventing a plan.
    """
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or getattr(terrain, "cfg", None) is None:
        return None
    course_cfg = terrain.cfg.terrain_generator.sub_terrains.get("rough_plum_blossom")
    centers = getattr(course_cfg, "block_centers", None) if course_cfg is not None else None
    if not centers or not hasattr(course_cfg, "block_size_range"):
        return None
    offsets = WHEEL_OFFSETS
    upper = [(float(x), float(y)) for x, y in centers if float(y) > 0.0]
    lower = [(float(x), float(y)) for x, y in centers if float(y) < 0.0]
    rows = int(terrain.terrain_origins.shape[0])
    lo, hi = float(course_cfg.block_size_range[0]), float(course_cfg.block_size_range[1])
    table = torch.zeros((rows, bins), dtype=torch.long, device=env.device)
    for row in range(rows):
        difficulty = row / float(max(rows - 1, 1))
        half = (lo + difficulty * (hi - lo)) * 0.5

        def on_cube(fx: float, fy: float) -> bool:
            group = upper if fy > 0.0 else lower
            return any(abs(fx - cx) <= half for cx, _ in group)

        for index in range(bins):
            body_x = index / float(bins - 1) * 2.7
            planted = {
                name
                for name, (dx, dy) in offsets.items()
                if on_cube(body_x + dx, dy)
            }
            if {"FL", "RR"} <= planted:
                table[row, index] = 1
            elif {"FR", "RL"} <= planted:
                table[row, index] = 2
    return table


def plum_gait_plan(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Penalize deviating from the precomputed reference diagonal.

    ``plum_support_pair`` penalizes the *absence* of any complete diagonal; this
    term additionally prefers the one the reference plan asks for at the current
    body position, which is the soft form of "compute the crossing, then let the
    policy learn it" (arXiv 2403.03639, ContactNet).  It is progress-indexed, not
    time-indexed, so the policy may go as slowly as it needs.  Where the plan has
    no static configuration (0 for 79.8% of the competition course) the term pays
    nothing and constrains nothing.

    A four-wheel stance completes both diagonals, so it satisfies the plan
    whichever one it asks for.  The old implementation collapsed the two
    diagonals into a single label (checking FL+RR first), which reported a
    mismatch for a stable four-contact stance whenever the plan asked for FR+RL.
    """
    zeros = torch.zeros(env.num_envs, device=env.device)
    mask = obstacle_row_mask(env, difficulty_threshold)
    if not bool(mask.any()):
        return zeros
    table = getattr(env, "_plum_gait_plan_table", None)
    if table is None:
        table = _build_gait_plan_table(env)
        if table is None:
            return zeros
        env._plum_gait_plan_table = table
    contact = _contact_flags(env, sensor_name) > 0.0
    if contact.shape[1] < 4:
        return zeros
    index = {name: i for i, name in enumerate(GO2W_FOOT_BODY_NAMES)}
    diagonal_a = contact[:, index["FL_foot"]] & contact[:, index["RR_foot"]]
    diagonal_b = contact[:, index["FR_foot"]] & contact[:, index["RL_foot"]]

    asset: Entity = env.scene["robot"]
    body_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    bins = table.shape[1]
    column = (body_x / 2.7 * (bins - 1)).round().long().clamp(0, bins - 1)
    row = env.scene.terrain.terrain_levels.long().clamp(0, table.shape[0] - 1)
    expected = table[row, column]
    satisfied = ((expected == 1) & diagonal_a) | ((expected == 2) & diagonal_b)
    mismatch = (expected > 0) & ~satisfied
    return mismatch.float() * mask.float()


def plum_support_pair(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Penalize a stance that has no *complete* diagonal support pair.

    On this course only a diagonal pair can be planted together, so the reference
    gait is a diagonal alternation -- the same structure a trot has.  Expressing
    it as a penalty for the missing structure (rather than a bonus for having it)
    is what keeps a static four-wheel stance from farming the term: both diagonals
    are complete when all four wheels are in contact, so that stance pays nothing.

    Measured feasibility of a complete diagonal pair along the course: 75.8% of the
    advance at 0.552 m cubes, 64.7% at 0.376 m, but only 24.7% at the competition
    0.200 m.  That is why a phase clock (trot timing) is *not* used here -- at the
    real geometry there is no statically valid reference to march to, while the
    structural preference is always meaningful.
    """
    contact = _contact_flags(env, sensor_name) > 0.0
    if contact.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    index = {name: i for i, name in enumerate(GO2W_FOOT_BODY_NAMES)}
    fl, fr = index["FL_foot"], index["FR_foot"]
    rl, rr = index["RL_foot"], index["RR_foot"]
    diagonal_a = contact[:, fl] & contact[:, rr]      # FL + RR
    diagonal_b = contact[:, fr] & contact[:, rl]      # FR + RL
    missing_diagonal = ~(diagonal_a | diagonal_b)
    return missing_diagonal.float() * obstacle_row_mask(
        env, difficulty_threshold
    ).float()


def plum_lateral_center_hold(
    env: ManagerBasedRlEnv,
    scale_m: float = 0.02,
    difficulty_threshold: float = 0.0,
    min_body_x: float = 0.45,
) -> torch.Tensor:
    """Reward holding the course centreline while on the stepping stones.

    The competition course leaves a 0.20 m gutter between the two rows of cubes.
    With the wheels ``track / 2 = 0.142`` apart, the inner edge of each wheel is
    only 16 mm from that gutter, so the robot has to straddle the centreline and
    stay on it: left wheels on the upper row, right wheels on the lower row.  No
    published stepping-stone course is this tight (BeamDojo uses 1.0 m stones), so
    this constraint has to be rewarded explicitly.

    ``min_body_x`` gates the term to the course itself: ``body_x = 0`` is the
    entry-platform centre and the platform edge is at 0.40 m, so without the gate
    the robot collected this reward for standing still before it ever touched a
    stone.  With ``scale_m = 0.02`` the raw value is ~0.30 at the centreline,
    ~0.11 at 2 cm and ~0.03 at 5 cm of lateral error (weight 0.10 in the task
    config).
    """
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or getattr(env.scene, "env_origins", None) is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Entity = env.scene["robot"]
    lateral = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    body_x = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    value = torch.exp(-torch.abs(lateral) / float(scale_m))
    on_course = (body_x > float(min_body_x)).float()
    return value * on_course * obstacle_row_mask(env, difficulty_threshold).float()


def plum_landing_softness(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    vz_scale: float = 0.15,
    difficulty_threshold: float = 0.0,
) -> torch.Tensor:
    """Reward placing a wheel on a cube softly instead of slamming it down.

    CN117008592A (quadruped crossing stepping stones) splits the landing into
    "touch the stone -> bend the legs to absorb -> keep the balance".  Our exit
    criterion only rejects a very hard landing (``max_vertical_speed``), so there
    is no positive incentive for the absorption phase.  This term supplies it:
    a gentle placement scores ~1, a 1 m/s impact scores ~0.0002.

    It complements ``plum_overhang`` -- that one governs *where* the wheel lands,
    this one governs *how* it lands.
    """
    placed = getattr(env, "_plum_foot_placed", None)
    if placed is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Entity = env.scene[asset_cfg.name]
    vertical_speed = torch.abs(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, 2])
    softness = torch.exp(-vertical_speed / float(vz_scale))
    return (softness * placed.float()).mean(dim=1) * obstacle_row_mask(
        env, difficulty_threshold
    ).float()


def plum_airborne_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    min_contacts: int = 2,
) -> torch.Tensor:
    """Penalize having fewer than ``min_contacts`` wheels on a surface.

    The stepping stones must be crossed one wheel at a time: lifting a single
    wheel leaves three in contact, and at the competition geometry a diagonal
    pair is the minimum support.  Two or fewer wheels in contact therefore means
    the robot is hopping or falling, neither of which can be aimed at a 0.20 m
    cube.  ``min_contacts = 2`` keeps the required two-point gait legal while
    making a flight phase expensive.
    """
    contact = _contact_flags(env, sensor_name)
    count = (contact > 0.0).float().sum(dim=1)
    deficit = torch.clamp(float(min_contacts) - count, min=0.0)
    return deficit * obstacle_row_mask(env).float()

