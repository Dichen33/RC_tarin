from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from minidog_native.terrain import bridge_length_tensor, bridge_row_ys

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _command_activity(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return command_norm, command_norm > command_threshold


def _stable_course_completion(
    env: ManagerBasedRlEnv,
    displacement: torch.Tensor,
    target: torch.Tensor,
    command: torch.Tensor,
    command_threshold: float,
    min_relative_height: float = 0.14,
    max_tilt: float = 0.70,
    min_forward_velocity: float = 0.02,
    max_vertical_velocity: float = 0.75,
    require_support: bool = True,
    require_collision_free: bool = True,
    require_timeout_free: bool = True,
) -> torch.Tensor:
    """Require a real, stable and continuing course exit.

    Position alone is not a success signal: a robot can cross a target while
    falling, sliding after a collision, or stopping immediately after a wall.
    The velocity checks make the event useful both for rewards and curriculum
    promotion without requiring privileged terrain labels in the actor.
    """
    asset: Entity = env.scene["robot"]
    relative_height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    tilt = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
    forward_velocity = asset.data.root_link_lin_vel_b[:, 0]
    vertical_velocity = torch.abs(asset.data.root_link_lin_vel_w[:, 2])
    collision_seen = _course_collision_history(env)
    support = _course_support_contact(env)
    terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
    if terminated is None:
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    timeouts = getattr(getattr(env, "termination_manager", None), "time_outs", None)
    if timeouts is None:
        timeouts = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (
        (displacement > target)
        & (command[:, 0] > command_threshold)
        & (relative_height > min_relative_height)
        & (tilt < max_tilt)
        & (forward_velocity > min_forward_velocity)
        & (vertical_velocity < max_vertical_velocity)
        & (
            (~collision_seen)
            if require_collision_free
            else torch.ones_like(collision_seen)
        )
        & (support if require_support else torch.ones_like(support))
        & ~terminated
        & ((~timeouts) if require_timeout_free else torch.ones_like(timeouts))
    )


def _contact_flags(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    found = (sensor.data.found > 0).float()
    while found.ndim > 2:
        found = torch.mean(found, dim=-1)
    if found.ndim == 1:
        found = found.unsqueeze(-1)
    return found


def _course_collision_free(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reject completion after body/leg contact with terrain.

    Wheel, foot and shank contacts are intentionally excluded: they are the
    normal support mode of Go2W, and a 0.086 m wheel can only climb a 0.10-0.32 m
    stepping stone or a 0.32 m stair by pressing the shank against its edge
    (the shank's lowest point sits at 0.078 m).  Including ``shank`` here vetoed
    almost every legitimate climb for the whole episode.  The shank is still
    penalized by the ``shank_collision`` reward term.  The listed sensors are
    base, trunk and thigh impacts, which indicate a fall or a collision-assisted
    crossing.
    """
    collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in (
        "base_ground_contact",
        "trunk_ground_touch",
        "thigh_ground_touch",
    ):
        try:
            flags = _contact_flags(env, sensor_name)
        except (KeyError, AttributeError):
            continue
        collision |= flags.reshape(env.num_envs, -1).any(dim=1)
    return ~collision


def _course_support_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return whether at least one foot/wheel supports the course exit."""
    try:
        contact = _contact_flags(env, "feet_ground_contact")
    except (KeyError, AttributeError):
        # Keep the term reusable with tasks that do not configure a contact
        # sensor. Go2W and the native obstacle tasks both configure it.
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return contact.reshape(env.num_envs, -1).any(dim=1)


def _course_collision_history(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Remember illegal body/leg contact for the whole episode.

    A final-frame check alone lets a robot hit a wall or the ditch, recover,
    and still receive a completion reward. The history is reset at episode
    boundaries by ``course_completion``.
    """
    current_collision = ~_course_collision_free(env)
    seen = getattr(env, "_course_collision_seen", None)
    if seen is None or seen.shape != current_collision.shape:
        seen = torch.zeros_like(current_collision)
    seen |= current_collision
    env._course_collision_seen = seen
    return seen


def _foot_body_state(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[Entity, torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    return asset, foot_pos, foot_vel


def _foot_pos_b(asset: Entity, foot_pos_w: torch.Tensor) -> torch.Tensor:
    rel_w = foot_pos_w - asset.data.root_link_pos_w[:, None, :]
    root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, rel_w.shape[1], -1)
    return quat_apply_inverse(root_quat, rel_w)


def _selected_joint_names(asset: Entity, asset_cfg: SceneEntityCfg) -> list[str]:
    if isinstance(asset_cfg.joint_ids, slice):
        return list(asset.joint_names[asset_cfg.joint_ids])
    return [asset.joint_names[i] for i in asset_cfg.joint_ids]


def _gait_phase(
    env: ManagerBasedRlEnv,
    command_norm: torch.Tensor,
    active: torch.Tensor,
    base_frequency: float,
    max_frequency: float,
    speed_scale: float,
) -> torch.Tensor:
    moving_frequency = torch.clamp(
        base_frequency + speed_scale * command_norm,
        min=base_frequency,
        max=max_frequency,
    )
    frequency = torch.where(active, moving_frequency, torch.zeros_like(command_norm))
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=command_norm.dtype)
    return (env.episode_length_buf.float() * env.step_dt * frequency + env_ids * 0.17320508) % 1.0


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def commanded_forward_motion(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.05,
    min_command: float = 0.08,
) -> torch.Tensor:
    """Reward actual forward motion when a positive forward command is active."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    target = torch.clamp(command[:, 0], min=min_command)
    active = command[:, 0] > command_threshold
    forward_velocity = torch.clamp(asset.data.root_link_lin_vel_b[:, 0], min=0.0)
    progress = torch.clamp(forward_velocity / target, min=0.0, max=1.0)
    return progress * active.float()


def commanded_forward_stall(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.05,
    min_command: float = 0.08,
) -> torch.Tensor:
    """Penalize standing still or moving backward under a forward command."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    target = torch.clamp(command[:, 0], min=min_command)
    active = command[:, 0] > command_threshold
    forward_velocity = torch.clamp(asset.data.root_link_lin_vel_b[:, 0], min=0.0)
    deficit = torch.clamp((target - forward_velocity) / target, min=0.0, max=1.0)
    return deficit * active.float()


def course_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    success_distance: float = 2.7,
    gap_success_distance: float = 3.35,
    wall_success_distance: float = 4.80,
    default_success_distance: float = 4.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Provide a small dense shaping signal for progress along the course."""
    asset: Entity = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    target = torch.full_like(displacement, default_success_distance)
    terrain = getattr(env.scene, "terrain", None)
    if terrain is not None and terrain.terrain_types is not None:
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        if "rough_plum_blossom" in names:
            target = torch.where(
                terrain.terrain_types == names.index("rough_plum_blossom"),
                torch.full_like(target, success_distance),
                target,
            )
        if "flat_gap" in names:
            target = torch.where(
                terrain.terrain_types == names.index("flat_gap"),
                torch.full_like(target, gap_success_distance),
                target,
            )
        if "flat_high_wall" in names:
            target = torch.where(
                terrain.terrain_types == names.index("flat_high_wall"),
                torch.full_like(target, wall_success_distance),
                target,
            )
    progress = torch.clamp(displacement / torch.clamp(target, min=1.0e-6), min=0.0, max=1.0)
    return progress * (command[:, 0] > command_threshold).float()


def _on_bridge_mask(
    foot_x: torch.Tensor,
    foot_y: torch.Tensor,
    block_size: torch.Tensor,
    bridge: torch.Tensor,
    bridge_rows: tuple[float, float],
    platform_half: float,
    exit_center_x: float,
) -> torch.Tensor:
    """Wheel positions that sit on an entry/exit wheel-line bridge.  ``[env, foot]``.

    A bridge is flush with the platform tops but is not a stone: without this mask
    every sole sample of a wheel rolling across it fell outside the cube layout, so
    ``plum_overhang`` punished a fully supported wheel, and near the bridge's inner
    end the loose cube-footprint test passed, so the foothold credit went to a stone
    the wheel had not reached.  The bridge only covers one cube row in y (which row
    is derived from the layout by ``bridge_row_ys``), so the other wheel line keeps
    its real slot.  5 mm of slack on each seam: a wheel straddling the
    platform/bridge or bridge/cube boundary is not reliably on either surface.
    """
    entry_y, exit_y = bridge_rows
    row_half = block_size[:, None] / 2.0 + 0.005
    length = bridge[:, None]
    # A faded-out bridge (the hard rows) must mask nothing, not even the 5 mm of
    # seam slack around the platform edge.
    present = length > 1.0e-4
    return present & (
        (
            (foot_x > platform_half - 0.005)
            & (foot_x < platform_half + length + 0.005)
            & (torch.abs(foot_y - entry_y) <= row_half)
        )
        | (
            (foot_x < exit_center_x - platform_half + 0.005)
            & (foot_x > exit_center_x - platform_half - length - 0.005)
            & (torch.abs(foot_y - exit_y) <= row_half)
        )
    )


class plum_foothold:
    """Track real wheel placement on the six stepping-stone top surfaces.

    The terrain generator keeps the course layout fixed in terrain-local
    coordinates. We combine those coordinates with the foot contact sensor and
    the wheel axle positions, so a forward walk on the lower floor cannot count
    as a successful plum-blossom crossing.  The wheel radius converts the axle
    position reported by the foot body into a contact height.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        del cfg
        self.visited = torch.zeros((env.num_envs, 6), dtype=torch.bool, device=env.device)
        self.blocks_completed = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self.course_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Same crossing event, gated on ``active_course`` like the dense terms.
        # Every row now generates a real course (``course_start_difficulty`` is
        # 0.0), so this only excludes envs that are not on a plum column.
        self.course_rewarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Dense placement signals: how many wheels are properly resting on a
        # stone top, and how many are jammed against a stone's side face.  At the
        # competition geometry only two wheels can be on stones at once, so the
        # skill to learn is lifting a wheel and placing it accurately -- these
        # two counters are what the placement rewards read.
        self.wheels_on_stone = torch.zeros(env.num_envs, device=env.device)
        self.edge_hits = torch.zeros(env.num_envs, device=env.device)
        # Per-foot placement flags (no stability filter), used by the landing
        # softness term: the patent for quadruped stepping-stone crossing splits
        # the landing into "touch the stone -> bend the legs to absorb -> keep the
        # balance", so the vertical speed at the moment of placement is a signal
        # of its own.  ``hits`` cannot be reused here because it already filters
        # on a small vertical speed, which would make the reward trivially 1.
        self.foot_placed = torch.zeros((env.num_envs, 4), dtype=torch.bool, device=env.device)
        env._plum_foot_placed = self.foot_placed
        # BeamDojo-style sole sampling: four points around the wheel contact patch
        # (half the wheel width sideways, 2 cm fore/aft).  The offsets are applied
        # in the patch frame, which is world-aligned here; with +/-0.026 m offsets
        # the yaw error is negligible for the 0.20 m cubes.
        self.foot_samples = torch.tensor(
            ((-0.02, -0.026), (-0.02, 0.026), (0.02, -0.026), (0.02, 0.026)),
            dtype=torch.float,
            device=env.device,
        )
        self.overhang = torch.zeros(env.num_envs, device=env.device)
        env._plum_wheels_on_stone = self.wheels_on_stone
        env._plum_edge_hits = self.edge_hits
        env._plum_overhang = self.overhang
        self.exit_stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.block_contact_steps = torch.zeros((env.num_envs, 6), dtype=torch.long, device=env.device)
        self.block_hold_steps = 2  # 40 ms prevents one-frame grazing contacts.
        self.exit_hold_steps = 3  # 60 ms at the 50 Hz policy rate.
        env._plum_course_success = self.course_success
        env._plum_course_rewarded = self.course_rewarded
        env._plum_blocks_completed = self.blocks_completed
        # Coordinates relative to the entrance-platform origin.  Read them from
        # the terrain cfg (single source of truth) so a course-layout change can
        # never desynchronise the foothold check from the generated blocks.  The
        # fallback has to be the same layout, not a mirror of it: row A (first cube
        # 0.6 m) is the *negative*-y row.
        centers = ((0.6, -0.2), (1.2, -0.2), (1.8, -0.2),
                   (0.9, 0.2), (1.5, 0.2), (2.1, 0.2))
        terrain = getattr(getattr(env, "scene", None), "terrain", None)
        course_cfg = None
        if terrain is not None and getattr(terrain, "cfg", None) is not None:
            course_cfg = terrain.cfg.terrain_generator.sub_terrains.get("rough_plum_blossom")
        if course_cfg is not None and getattr(course_cfg, "block_centers", None):
            centers = tuple(course_cfg.block_centers)
        self.block_centers = torch.tensor(
            centers,
            dtype=torch.float,
            device=env.device,
        )
        # Same-row cube pitch and row separation, read from the single layout
        # source instead of hard-coding 0.60 / 0.40.  The bootcamp row is a
        # continuous slab only because both distances are <= its cube size.
        self.same_row_pitch = float(self.block_centers[1, 0] - self.block_centers[0, 0])
        self.row_separation = float(
            torch.abs(self.block_centers[3, 1] - self.block_centers[0, 1])
        )
        # Entry/exit wheel-line bridges (see ``bridge_fade_end`` in the terrain
        # cfg).  A bridge is flush with the platform tops but is not a stone, so
        # its footprint is cut out of the course span below.  ``None`` means the
        # course has no bridges and the span stays as the terrain builder drew it.
        self.bridge_rows: tuple[float, float] | None = None
        self.course_cfg = course_cfg
        if course_cfg is not None and float(getattr(course_cfg, "bridge_fade_end", 0.0)) > 0.0:
            self.bridge_rows = bridge_row_ys(course_cfg)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        success_distance: float = 2.7,
        command_threshold: float = 0.05,
        # Fallback geometry, only used when the terrain has no ``rough_plum_blossom``
        # sub-terrain (the runtime path reads the terrain cfg, which is the single
        # source of truth).  Kept equal to the P0 bootcamp ladder -- 0.60 -> 0.20 m
        # cubes, 0.10 -> 0.20 platform, course present from row 0 -- so a failed
        # lookup can never silently reinstate the old drifted 0.36/0.28 values.
        course_start_difficulty: float = 0.0,
        block_size_range: tuple[float, float] = (0.60, 0.20),
        block_height_range: tuple[float, float] = (0.10, 0.20),
        platform_height: float = 0.20,
        top_tolerance: float = 0.055,
        max_vertical_speed: float = 0.50,
        wheel_radius: float = 0.0,
        require_all_stones: bool = True,
    ) -> torch.Tensor:
        terrain = getattr(env.scene, "terrain", None)
        names = [] if terrain is None or terrain.cfg.terrain_generator is None else list(
            terrain.cfg.terrain_generator.sub_terrains.keys()
        )
        if terrain is None or "rough_plum_blossom" not in names:
            self.course_success.zero_()
            self.blocks_completed.zero_()
            return torch.zeros(env.num_envs, device=env.device)

        # The terrain config owns the course geometry. Reading it here keeps the
        # reward and the generated stones in sync; the duplicated constants in
        # the task config had already drifted apart once.
        course_cfg = terrain.cfg.terrain_generator.sub_terrains.get("rough_plum_blossom")
        platform_width = 0.80
        exit_center_x = float(success_distance)
        if course_cfg is not None:
            block_size_range = tuple(course_cfg.block_size_range)
            block_height_range = tuple(course_cfg.block_height_range)
            platform_height_base = float(course_cfg.platform_height)
            course_start_difficulty = float(course_cfg.course_start_difficulty)
            platform_width = float(getattr(course_cfg, "platform_width", platform_width))
            exit_center_x = float(getattr(course_cfg, "exit_center_x", exit_center_x))
        else:
            platform_height_base = float(platform_height)

        asset: Entity = env.scene[asset_cfg.name]
        foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
        contact = _contact_flags(env, sensor_name)[:, : foot_pos.shape[1]] > 0.0
        plum_mask = terrain.terrain_types == names.index("rough_plum_blossom")
        num_rows = max(int(terrain.terrain_origins.shape[0]), 2)
        difficulty = terrain.terrain_levels.float() / float(num_rows - 1)
        active_course = plum_mask & (difficulty >= course_start_difficulty)

        # Same (easy, hard) convention as the terrain builder, via the shared
        # helper: the reward used to interpolate the cube size in the opposite
        # direction, so it scored placements against a footprint the terrain did
        # not have.  It also re-implemented the ramp, so a shaped ramp
        # (``size_ramp_exponent``) would have desynchronised terrain and reward
        # exactly the way the reversed copy did before.
        from minidog_native.terrain import course_geometry_at as _course_geometry_at

        block_size, block_height, platform_height = _course_geometry_at(
            course_cfg if course_cfg is not None else type("_G", (), {
                "block_size_range": block_size_range,
                "block_height_range": block_height_range,
                "platform_height": platform_height_base,
                "size_ramp_exponent": 1.0,
            })(),
            difficulty,
        )
        foot_pos_local = foot_pos[:, :, :2] - env.scene.env_origins[:, None, :2]
        delta = torch.abs(foot_pos_local[:, :, None, :] - self.block_centers[None, None, :, :])
        # [env, foot, block, xy] broadcasting.  The explicit fourth axis is
        # important when more than one environment is evaluated in parallel.
        half_size = block_size[:, None, None, None] / 2.0 + 0.06
        # The platforms are level with the stone tops on every row, so the loose
        # footprint test alone counted a wheel resting on the entrance platform
        # as a placement on the first stone -- ``blocks_completed`` reported 1 for
        # a robot that had never left the spawn platform.  Stones only exist in
        # the open span between the two platforms.
        platform_half = platform_width / 2.0
        course_span = (
            (foot_pos_local[:, :, 0] > platform_half + 0.02)
            & (foot_pos_local[:, :, 0] < exit_center_x - platform_half - 0.02)
        )
        # A wheel on an entry/exit bridge is on a platform, not on a stone; see
        # ``_on_bridge_mask`` for what the tests would score without this.
        if self.bridge_rows is not None:
            course_span = course_span & ~_on_bridge_mask(
                foot_pos_local[:, :, 0],
                foot_pos_local[:, :, 1],
                block_size,
                bridge_length_tensor(self.course_cfg, block_size, difficulty),
                self.bridge_rows,
                platform_half,
                exit_center_x,
            )
        inside = (
            torch.all(delta <= half_size, dim=-1) & course_span[:, :, None]
        )
        relative_foot_z = foot_pos[:, :, 2][:, :, None] - env.scene.env_origins[:, 2][:, None, None]
        # ``block_height`` and ``platform_height`` come from ``course_geometry_at``
        # above, i.e. the same call that built the terrain.  Recomputing them here
        # (as this block used to) is how the reward drifted from the geometry twice.
        # The platform is *ramped* too (``0.5 + 0.5 * d`` in the terrain builder).
        # Using the un-ramped 0.20 m here put the expected stone top up to 0.10 m
        # too low on the easy rows, which is larger than ``top_tolerance`` -- so
        # "on top of a cube" was never detected on rows 0-1, where half the plum
        # episodes run.
        block_top_relative = block_height - platform_height
        # ``foot_pos`` is the wheel *axle* (the foot body origin sits on the
        # wheel axis), so a wheel standing on a surface has its axle one wheel
        # radius above that surface.  Comparing the axle with the block top is
        # off by WHEEL_RADIUS, which is larger than ``top_tolerance``: no stone
        # could ever be registered.
        contact_z = relative_foot_z - wheel_radius
        on_top = torch.abs(
            contact_z - block_top_relative[:, None, None]
        ) <= top_tolerance
        foot_vertical_speed = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, 2]
        stable_contact = torch.abs(foot_vertical_speed) <= max_vertical_speed
        hits = (
            inside
            & on_top
            & stable_contact[:, :, None]
            & contact[:, :, None]
            & active_course[:, None, None]
        )
        block_contact = hits.any(dim=1)
        # Dense placement counters (see __init__).  ``hits`` is [env, foot, block];
        # a wheel counts as placed when it rests on some stone top with a stable
        # contact, and as an edge hit when it is inside the stone footprint but
        # clearly below the top face -- i.e. it clanged into the side.
        self.wheels_on_stone[:] = hits.any(dim=2).float().sum(dim=1)
        placed = inside & on_top & contact[:, :, None] & active_course[:, None, None]
        self.foot_placed[:] = placed.any(dim=2)
        # ``contact_z`` and ``block_top_relative`` are both [env, foot, 1], so the
        # comparison stays 3-D and broadcasts against ``inside`` [env, foot, block].
        below_top = contact_z < (block_top_relative[:, None, None] - top_tolerance)
        edge = inside & below_top & active_course[:, None, None]
        self.edge_hits[:] = edge.any(dim=2).float().sum(dim=1)

        # BeamDojo eq. (2): sample points on the sole of every contacting foot and
        # count those that are *not* over a stone.  A wheel that half-overhangs a
        # cube still satisfies the centre test above, but it will tip; the
        # overhang count is the signal that penalizes it.  Sampling is analytic
        # against the known block layout, so no extra ray casts are needed.
        if self.foot_samples is not None:
            samples = foot_pos_local[:, :, None, :] + self.foot_samples[None, None, :, :]
            sample_delta = torch.abs(
                samples[:, :, :, None, :] - self.block_centers[None, None, None, :, :]
            )
            # ``half_size`` (block/2 + 6 cm) is deliberately loose for the
            # centre test; the sole samples must sit *inside* the cube instead,
            # so use a tight margin here or a wheel hanging off a 0.20 m cube
            # would still count as covered.
            sample_half = (block_size[:, None, None, None] / 2.0 - 0.01).clamp_min(0.02)[..., None]
            sample_inside = torch.all(sample_delta <= sample_half, dim=-1)
            covered = sample_inside.any(dim=-1)
            overhang = (~covered).float().sum(dim=2) * contact.float()
            # A wheel standing on either platform is not overhanging a stone;
            # without this gate every episode opened with a constant penalty
            # (four wheels x four samples) for the pose the robot spawns in.
            overhang = overhang * course_span.float()
            self.overhang[:] = overhang.sum(dim=1) * active_course.float()
        self.block_contact_steps = torch.where(
            block_contact,
            self.block_contact_steps + 1,
            torch.zeros_like(self.block_contact_steps),
        )
        confirmed_hits = self.block_contact_steps >= self.block_hold_steps
        # A row certifies progress only if the wheel path has a real gap.  On the
        # 0.60 m bootcamp row the cubes touch inside each row and the two rows
        # overlap, so the course is one continuous slab: every cube position is
        # always covered and visiting them would certify a trivial 6, which would
        # promote the difficulty cap for free.  The old test only looked at the
        # central gutter and therefore still credited the 0.40 m rows, whose
        # same-row gaps (0.20 m) are real.  Both gaps are checked here.
        same_row_gap = self.same_row_pitch - block_size
        row_gutter = self.row_separation - block_size
        has_step = (same_row_gap > 0.02) | (row_gutter > 0.02)
        confirmed_hits = confirmed_hits & has_step[:, None]
        solid_row = ~has_step
        newly_visited = confirmed_hits & ~self.visited
        self.visited |= confirmed_hits
        self.blocks_completed[:] = self.visited.float().sum(dim=1)

        displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
        command = env.command_manager.get_command(command_name)
        target = torch.full_like(displacement, success_distance)
        exit_success = _stable_course_completion(
            env,
            displacement,
            target,
            command,
            command_threshold,
            # Falling off the course is already a termination, so a stumble that
            # the robot recovered from must not disqualify the crossing.
            require_collision_free=False,
            # A careful crossing ends with the robot standing still on the exit
            # platform, and 2.7 m at ~0.13 m/s takes 20 s of the 30 s episode, so
            # the arrival frequently coincides with the timeout.  Neither "not
            # still rolling" nor "the clock ran out" means the crossing failed:
            # measured 37.7% of plum episodes reached the exit distance yet the
            # success channel reported 0.
            min_forward_velocity=0.0,
            require_timeout_free=False,
        )
        # Rows below ``course_start_difficulty`` contain no course at all: the
        # terrain generator returns a plain rough patch there (its own "warm-up
        # row").  Requiring ``active_course`` here made every such episode an
        # automatic failure.  Measured on a 3000-iteration plum-only run: ~84% of
        # plum episodes sat on such a row, which capped the family success rate
        # near 0.16 and pinned the difficulty cap at 1, so the policy never saw a
        # real course row.  Without the course the route requirement is already
        # waived below, so a stable forward exit is the right score there.  The
        # threshold is 0.0 now, so this branch is vacuous, but the exit check
        # stays unconditioned: raising the threshold again must not re-create the
        # automatic-failure deadlock.
        exit_candidate = exit_success & plum_mask
        self.exit_stable_steps = torch.where(
            exit_candidate,
            self.exit_stable_steps + 1,
            torch.zeros_like(self.exit_stable_steps),
        )
        stable_exit = self.exit_stable_steps >= self.exit_hold_steps
        # The six stones form two staggered rows of three.  Requiring all six
        # forces a serpentine path across both rows, which made the success rate
        # identically zero; a legitimate crossing follows one complete row.
        route_done = self.visited.all(dim=1)
        if not require_all_stones:
            route_done = self.visited[:, :3].all(dim=1) | self.visited[:, 3:].all(dim=1)
        # The bootcamp row is one continuous slab: there is nothing to visit, so a
        # stable arrival at the exit is the only meaningful success check.  This
        # is what makes row 0 learnable instead of an automatic failure.
        route_done = route_done | solid_row
        self.course_success[:] = plum_mask & torch.where(
            active_course,
            route_done,
            torch.ones_like(active_course),
        ) & stable_exit
        # Reward channel: the same event, minus the course-free warm-up rows.
        self.course_rewarded[:] = plum_mask & route_done & active_course & stable_exit
        # Keep the completed event available to curriculum callbacks before
        # RewardManager.reset() clears this term's episode state.
        env._obstacle_course_success = getattr(
            env, "_obstacle_course_success", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
        env._obstacle_course_success[:] = torch.where(
            plum_mask, self.course_success, env._obstacle_course_success
        )
        return newly_visited.float().sum(dim=1)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.visited[env_ids] = False
        self.blocks_completed[env_ids] = 0.0
        self.course_success[env_ids] = False
        self.block_contact_steps[env_ids] = 0
        self.exit_stable_steps[env_ids] = 0


class course_completion:
    """Give one reward when an active episode reaches the exit platform."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        del cfg
        self.completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.env = env
        self.exit_stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.exit_hold_steps = 5

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        success_distance: float = 2.7,
        gap_success_distance: float = 3.35,
        wall_success_distance: float = 4.80,
        default_success_distance: float = 4.0,
        command_threshold: float = 0.05,
        min_forward_velocity: float = 0.02,
        max_vertical_velocity: float = 0.75,
    ) -> torch.Tensor:
        asset: Entity = env.scene["robot"]
        command = env.command_manager.get_command(command_name)
        displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
        target = torch.full_like(displacement, default_success_distance)
        terrain = getattr(env.scene, "terrain", None)
        names: list[str] = []
        if terrain is not None and terrain.terrain_types is not None:
            names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
            if "rough_plum_blossom" in names:
                target = torch.where(
                    terrain.terrain_types == names.index("rough_plum_blossom"),
                    torch.full_like(target, success_distance),
                    target,
                )
            if "flat_gap" in names:
                target = torch.where(
                    terrain.terrain_types == names.index("flat_gap"),
                    torch.full_like(target, gap_success_distance),
                    target,
                )
            if "flat_high_wall" in names:
                target = torch.where(
                    terrain.terrain_types == names.index("flat_high_wall"),
                    torch.full_like(target, wall_success_distance),
                    target,
                )
        success = _stable_course_completion(
            env,
            displacement,
            target,
            command,
            command_threshold,
            min_forward_velocity=min_forward_velocity,
            max_vertical_velocity=max_vertical_velocity,
        )
        plum_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        plum_course_success = None
        if terrain is not None and terrain.terrain_types is not None and "rough_plum_blossom" in names:
            plum_mask = terrain.terrain_types == names.index("rough_plum_blossom")
            # Prefer the reward channel: it withholds the bonus on rows that
            # have no stones, so the completion reward cannot be farmed by
            # driving across the empty warm-up row.
            plum_course_success = getattr(env, "_plum_course_rewarded", None)
            if plum_course_success is None:
                plum_course_success = getattr(env, "_plum_course_success", None)
            if plum_course_success is not None:
                success = torch.where(plum_mask, plum_course_success, success)
        self.exit_stable_steps = torch.where(
            success,
            self.exit_stable_steps + 1,
            torch.zeros_like(self.exit_stable_steps),
        )
        stable_success = self.exit_stable_steps >= self.exit_hold_steps
        # ``plum_foothold`` already requires the six stones and its own
        # five-step stable exit. Do not add a second 100 ms hold window.
        if plum_course_success is not None:
            stable_success = torch.where(plum_mask, plum_course_success, stable_success)
        new_success = stable_success & ~self.completed
        self.completed |= stable_success
        # Curriculum terms run during reset, before this reward object's reset
        # method. Preserve the episode result for type-specific promotion.
        env._obstacle_course_success = self.completed.clone()
        return new_success.float()

    def reset(self, env_ids: torch.Tensor) -> None:
        self.completed[env_ids] = False
        self.exit_stable_steps[env_ids] = 0
        if hasattr(self.env, "_course_collision_seen"):
            self.env._course_collision_seen[env_ids] = False
        if hasattr(self.env, "_obstacle_course_success"):
            self.env._obstacle_course_success[env_ids] = False


class gap_crossing:
    """One-time reward when an active policy clears the ditch exit."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        del cfg
        self.completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.env = env
        self.exit_stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.exit_hold_steps = 5

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        gap_success_distance: float = 3.35,
        command_threshold: float = 0.05,
        min_forward_velocity: float = 0.02,
        max_vertical_velocity: float = 0.75,
    ) -> torch.Tensor:
        asset: Entity = env.scene["robot"]
        command = env.command_manager.get_command(command_name)
        terrain = getattr(env.scene, "terrain", None)
        if terrain is None or terrain.terrain_types is None:
            return torch.zeros(env.num_envs, device=env.device)
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        if "flat_gap" not in names:
            return torch.zeros(env.num_envs, device=env.device)

        displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
        is_gap = terrain.terrain_types == names.index("flat_gap")
        target = torch.full_like(displacement, gap_success_distance)
        success = is_gap & _stable_course_completion(
            env,
            displacement,
            target,
            command,
            command_threshold,
            min_forward_velocity=min_forward_velocity,
            max_vertical_velocity=max_vertical_velocity,
        )
        self.exit_stable_steps = torch.where(
            success,
            self.exit_stable_steps + 1,
            torch.zeros_like(self.exit_stable_steps),
        )
        stable_success = self.exit_stable_steps >= self.exit_hold_steps
        new_success = stable_success & ~self.completed
        self.completed |= stable_success
        # Keep the event available to the curriculum even when this focused
        # gap term is the only obstacle-specific reward that is enabled.
        obstacle_success = getattr(
            env,
            "_obstacle_course_success",
            torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        )
        env._obstacle_course_success = torch.where(is_gap, self.completed, obstacle_success)
        return new_success.float()

    def reset(self, env_ids: torch.Tensor) -> None:
        self.completed[env_ids] = False
        self.exit_stable_steps[env_ids] = 0
        if hasattr(self.env, "_obstacle_course_success"):
            self.env._obstacle_course_success[env_ids] = False


def track_yaw_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)


def base_height_l2(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg | None = None,
    support_sensor: str | None = None,
    foot_asset_cfg: SceneEntityCfg | None = None,
    wheel_radius: float = 0.0,
    height_error_clip: float = 0.5,
) -> torch.Tensor:
    """Penalize deviation from the target base height above the ground.

    ``support_sensor`` plus ``foot_asset_cfg`` switch the reference plane from
    the spawn origin to the local support plane: the lowest wheel that is
    currently in contact, offset by ``wheel_radius``.  On stairs, stepping
    stones and walls the spawn plane stays far below the robot, so the fixed
    reference punished climbing itself.  While airborne no wheel supports the
    robot and the reference falls back to the spawn plane.  ``height_error_clip``
    keeps a single bad state (for example a free fall) from dominating the
    return.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    origins_z = getattr(env.scene, "env_origins", None)
    if origins_z is None:
        ground_z = torch.zeros(env.num_envs, device=env.device)
    else:
        ground_z = env.scene.env_origins[:, 2]
    if support_sensor is not None and foot_asset_cfg is not None:
        try:
            contact = _contact_flags(env, support_sensor)
        except (KeyError, AttributeError):
            contact = None
        if contact is not None:
            foot_asset: Entity = env.scene[foot_asset_cfg.name]
            foot_z = foot_asset.data.body_link_pos_w[:, foot_asset_cfg.body_ids, 2]
            count = min(int(foot_z.shape[1]), int(contact.shape[1]))
            if count > 0:
                support = foot_z[:, :count] - wheel_radius
                in_contact = contact[:, :count] > 0.0
                support = torch.where(
                    in_contact,
                    support,
                    torch.full_like(support, float("inf")),
                )
                lowest = support.amin(dim=1)
                ground_z = torch.where(torch.isfinite(lowest), lowest, ground_z)
    error = asset.data.root_link_pos_w[:, 2] - ground_z - target_height
    if height_error_clip > 0.0:
        error = torch.clamp(error, -height_error_clip, height_error_clip)
    return torch.square(error)


def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def safe_base_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
    from mjlab.envs.mdp.observations import base_lin_vel

    value = base_lin_vel(env)
    return torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)


def safe_height_scan(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    scan_update_hz: float = 0.0,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
) -> torch.Tensor:
    """Return a finite height scan, optionally emulating an ODIN1 scan stream.

    The actor can use this function with ``scan_update_hz=10`` to hold a
    noisy scan between sensor updates, matching the DTOF update rate.  The
    critic keeps the defaults and therefore receives the instantaneous clean
    scan.  Cache names include the corruption settings so actor and critic do
    not accidentally share a noisy tensor.
    """
    from mjlab.envs.mdp.observations import height_scan

    # Actor and critic observation groups request the same simulated scan in
    # sequence.  Cache the raw scan once per environment step, then apply the
    # actor-only update rate/noise/dropout below without changing the sensor
    # resolution or the deployment observation contract.
    raw_cache_name = f"_minidog_height_scan_raw_{sensor_name}"
    raw_step_name = f"{raw_cache_name}_step"
    step_counter = int(getattr(env, "common_step_counter", -1))
    raw_cache = getattr(env, raw_cache_name, None)
    cached_step = getattr(env, raw_step_name, None)
    if (
        raw_cache is None
        or raw_cache.device != env.device
        or cached_step != step_counter
    ):
        raw_cache = torch.nan_to_num(height_scan(env, sensor_name), nan=0.0, posinf=3.0, neginf=-3.0).clone()
        setattr(env, raw_cache_name, raw_cache)
        setattr(env, raw_step_name, step_counter)
    value = raw_cache
    noise_std = max(float(noise_std), 0.0)
    dropout_prob = min(max(float(dropout_prob), 0.0), 1.0)
    update_hz = max(float(scan_update_hz), 0.0)
    if update_hz <= 0.0 and noise_std <= 0.0 and dropout_prob <= 0.0:
        return value

    step_dt = max(float(getattr(env, "step_dt", 0.0)), 1.0e-6)
    update_interval = max(int(round(1.0 / (update_hz * step_dt))) if update_hz > 0.0 else 1, 1)
    cache_name = (
        f"_minidog_height_scan_cache_{sensor_name}_"
        f"{int(update_hz * 100):d}_{int(noise_std * 1000):d}_{int(dropout_prob * 100):d}"
    )
    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape != value.shape or cache.device != value.device:
        cache = value.clone()
        setattr(env, cache_name, cache)
        refresh = torch.ones(value.shape[0], dtype=torch.bool, device=value.device)
    else:
        episode_steps = getattr(env, "episode_length_buf", None)
        if episode_steps is None:
            refresh = torch.ones(value.shape[0], dtype=torch.bool, device=value.device)
        else:
            refresh = (episode_steps <= 0) | (episode_steps.remainder(update_interval) == 0)

    if torch.any(refresh):
        sampled = value
        if noise_std > 0.0:
            sampled = sampled + torch.randn_like(sampled) * noise_std
        if dropout_prob > 0.0:
            sampled = torch.where(
                torch.rand_like(sampled) < dropout_prob,
                # MuJoCo/ODIN1 no-return is represented by the clipped
                # max-range value, not a false depression in the terrain.
                torch.ones_like(sampled),
                sampled,
            )
        cache[refresh] = sampled[refresh]
    return cache


def joint_deviation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
        )
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(torch.square(q - q0), dim=1)


def stand_pose(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.03,
    hip_weight: float = 1.2,
    thigh_weight: float = 1.6,
    calf_weight: float = 1.6,
    velocity_weight: float = 0.03,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
        )
    _, active = _command_activity(env, command_name, command_threshold)
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = _selected_joint_names(asset, asset_cfg)
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    weights = []
    for name in joint_names:
        if "_hip_joint" in name:
            weights.append(hip_weight)
        elif "_thigh_joint" in name:
            weights.append(thigh_weight)
        elif "_calf_joint" in name:
            weights.append(calf_weight)
        else:
            weights.append(1.0)
    weight = torch.tensor(weights, device=env.device, dtype=q.dtype).unsqueeze(0)
    pose_cost = torch.mean(torch.square(q - q0) * weight, dim=1)
    vel_cost = torch.mean(torch.square(qd), dim=1)
    return (pose_cost + velocity_weight * vel_cost) * (~active).float()


def stand_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.04,
) -> torch.Tensor:
    contact = _contact_flags(env, sensor_name)
    _, active = _command_activity(env, command_name, command_threshold)
    if contact.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    missing_contact = 1.0 - torch.clamp(contact[:, :4], 0.0, 1.0)
    return torch.mean(missing_contact, dim=1) * (~active).float()


def stand_foot_geometry(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.04,
    min_width: float = 0.18,
    min_length: float = 0.18,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, active = _command_activity(env, command_name, command_threshold)
    asset, foot_pos_w, _ = _foot_body_state(env, asset_cfg)
    if foot_pos_w.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    foot_pos = _foot_pos_b(asset, foot_pos_w)

    front_width = torch.abs(foot_pos[:, 0, 1] - foot_pos[:, 1, 1])
    rear_width = torch.abs(foot_pos[:, 2, 1] - foot_pos[:, 3, 1])
    left_length = torch.abs(foot_pos[:, 0, 0] - foot_pos[:, 2, 0])
    right_length = torch.abs(foot_pos[:, 1, 0] - foot_pos[:, 3, 0])
    width_cost = 0.5 * (
        torch.square(torch.clamp(min_width - front_width, min=0.0) / max(min_width, 1.0e-6))
        + torch.square(torch.clamp(min_width - rear_width, min=0.0) / max(min_width, 1.0e-6))
    )
    length_cost = 0.5 * (
        torch.square(torch.clamp(min_length - left_length, min=0.0) / max(min_length, 1.0e-6))
        + torch.square(torch.clamp(min_length - right_length, min=0.0) / max(min_length, 1.0e-6))
    )
    pair_symmetry = (
        torch.square((foot_pos[:, 0, 0] - foot_pos[:, 1, 0]) / max(min_length, 1.0e-6))
        + torch.square((foot_pos[:, 2, 0] - foot_pos[:, 3, 0]) / max(min_length, 1.0e-6))
        + torch.square((foot_pos[:, 0, 1] - foot_pos[:, 2, 1]) / max(min_width, 1.0e-6))
        + torch.square((foot_pos[:, 1, 1] - foot_pos[:, 3, 1]) / max(min_width, 1.0e-6))
    )
    return (width_cost + 0.5 * length_cost + 0.15 * pair_symmetry) * (~active).float()


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.03,
    lin_vel_threshold: float = 0.04,
    yaw_vel_threshold: float = 0.06,
    height_tol: float = 0.02,
    tilt_tol: float = 0.12,
    target_height: float = 0.235,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
        )
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    inactive = (command_norm < command_threshold).float()
    joint_cost = joint_deviation_l2(env, asset_cfg)
    base_lin_vel = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    yaw_vel = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
    tilt_cost = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)

    if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
        ground_z = env.scene.env_origins[:, 2]
    else:
        ground_z = torch.zeros(env.num_envs, device=env.device)
    height_error = torch.abs(asset.data.root_link_pos_w[:, 2] - ground_z - target_height)

    stability_cost = (
        joint_cost
        + torch.relu(base_lin_vel - lin_vel_threshold)
        + 0.5 * torch.relu(yaw_vel - yaw_vel_threshold)
        + 2.0 * torch.relu(height_error - height_tol)
        + 0.8 * torch.relu(tilt_cost - tilt_tol)
    )
    stability_cost *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return stability_cost * inactive


def contact_fraction_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    found = (sensor.data.found > 0).float()
    while found.ndim > 2:
        found = torch.mean(found, dim=-1)
    return torch.mean(found, dim=1)


def contact_fraction_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return contact_fraction_reward(env, sensor_name).unsqueeze(-1)


def feet_contact_target_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.03,
    target_contact: float = 4.0,
    sigma: float = 0.75,
) -> torch.Tensor:
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    found = (sensor.data.found > 0).float()
    while found.ndim > 2:
        found = torch.mean(found, dim=-1)
    if found.ndim == 1:
        found = found.unsqueeze(-1)
    contact_count = torch.sum(found, dim=1)

    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    inactive = (command_norm < command_threshold).float()

    reward = torch.exp(-torch.square(contact_count - target_contact) / max(sigma, 1.0e-6) ** 2)
    return reward * inactive


def feet_gait_consistency(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.03,
    duty_factor: float = 0.55,
    transition_width: float = 0.08,
    base_frequency: float = 1.8,
    max_frequency: float = 2.8,
    speed_scale: float = 1.0,
) -> torch.Tensor:
    contact = _contact_flags(env, sensor_name)
    if contact.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)

    command_norm, active = _command_activity(env, command_name, command_threshold)
    phase = _gait_phase(env, command_norm, active, base_frequency, max_frequency, speed_scale)
    group_a_phase = phase
    group_b_phase = (phase + 0.5) % 1.0

    def stance_gate(local_phase: torch.Tensor) -> torch.Tensor:
        distance_to_edge = torch.minimum(local_phase, torch.abs(local_phase - duty_factor))
        raw = torch.clamp(distance_to_edge / max(transition_width, 1.0e-6), 0.0, 1.0)
        return torch.where(local_phase < duty_factor, raw, torch.zeros_like(raw))

    stance_a = stance_gate(group_a_phase)
    stance_b = stance_gate(group_b_phase)
    expected = torch.stack((stance_a, stance_b, stance_b, stance_a), dim=1)
    reward = 1.0 - torch.mean(torch.abs(contact[:, :4] - expected), dim=1)
    return torch.clamp(reward, min=0.0) * active.float()


def foot_clearance(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    target_height: float = 0.055,
    command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, active = _command_activity(env, command_name, command_threshold)
    contact = _contact_flags(env, sensor_name)
    _, foot_pos, foot_vel = _foot_body_state(env, asset_cfg)
    if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
        ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
    else:
        ground_z = torch.zeros(env.num_envs, 1, device=env.device)
    clearance = foot_pos[:, :, 2] - ground_z
    swing = 1.0 - contact[:, : clearance.shape[1]]
    moving_foot = torch.norm(foot_vel[:, :, :2], dim=2)
    error = torch.square(torch.clamp(target_height - clearance, min=0.0))
    return torch.mean(error * swing * moving_foot, dim=1) * active.float()


class foot_swing_height:
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg",
            SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True),
        )
        asset: Entity = env.scene[asset_cfg.name]
        body_ids = asset_cfg.body_ids
        if isinstance(body_ids, slice):
            num_feet = len(asset.body_names[body_ids])
        else:
            num_feet = len(body_ids)
        self.peak_heights = torch.zeros((env.num_envs, num_feet), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        command_name: str,
        target_height: float = 0.065,
        command_threshold: float = 0.03,
        asset_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        from mjlab.sensor import ContactSensor

        if asset_cfg is None:
            asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
        command_norm, active = _command_activity(env, command_name, command_threshold)
        del command_norm

        sensor: ContactSensor = env.scene[sensor_name]
        asset, foot_pos, _ = _foot_body_state(env, asset_cfg)
        del asset

        if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
            ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
        else:
            ground_z = torch.zeros(env.num_envs, 1, device=env.device)
        clearance = foot_pos[:, :, 2] - ground_z

        contact = _contact_flags(env, sensor_name)[:, : clearance.shape[1]]
        in_air = contact <= 0.0
        self.peak_heights = torch.where(in_air, torch.maximum(self.peak_heights, clearance), self.peak_heights)

        first_contact = sensor.compute_first_contact(dt=env.step_dt)[:, : clearance.shape[1]]
        normalized_error = self.peak_heights / max(target_height, 1.0e-6) - 1.0
        cost = torch.mean(torch.square(normalized_error) * first_contact.float(), dim=1) * active.float()

        self.peak_heights = torch.where(first_contact, torch.zeros_like(self.peak_heights), self.peak_heights)
        return cost

    def reset(self, env_ids: torch.Tensor) -> None:
        self.peak_heights[env_ids] = 0.0


def foot_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, active = _command_activity(env, command_name, command_threshold)
    contact = _contact_flags(env, sensor_name)
    _, _, foot_vel = _foot_body_state(env, asset_cfg)
    contact = contact[:, : foot_vel.shape[1]]
    slip_speed = torch.norm(foot_vel[:, :, :2], dim=2)
    return torch.mean(slip_speed * contact, dim=1) * active.float()


def leg_mirror(
    env: ManagerBasedRlEnv,
    command_name: str | None = None,
    command_threshold: float = 0.03,
    active_only: bool = False,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
            preserve_order=True,
        )
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_names = _selected_joint_names(asset, asset_cfg)
    if not joint_names:
        return torch.zeros(env.num_envs, device=env.device)
    name_to_index = {name: i for i, name in enumerate(joint_names)}

    pairs = (
        ("FL_hip_joint", "FR_hip_joint", -1.0),
        ("RL_hip_joint", "RR_hip_joint", -1.0),
        ("FL_thigh_joint", "FR_thigh_joint", 1.0),
        ("RL_thigh_joint", "RR_thigh_joint", 1.0),
        ("FL_calf_joint", "FR_calf_joint", 1.0),
        ("RL_calf_joint", "RR_calf_joint", 1.0),
        ("FL_hip_joint", "RL_hip_joint", 1.0),
        ("FR_hip_joint", "RR_hip_joint", 1.0),
    )
    costs = []
    for left, right, sign in pairs:
        if left in name_to_index and right in name_to_index:
            costs.append(torch.square(q[:, name_to_index[left]] - sign * q[:, name_to_index[right]]))
    if not costs:
        return torch.zeros(env.num_envs, device=env.device)
    reward = torch.mean(torch.stack(costs, dim=1), dim=1)
    if active_only and command_name is not None:
        _, active = _command_activity(env, command_name, command_threshold)
        reward = reward * active.float()
    return reward


def diagonal_leg_symmetry(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.03,
    velocity_weight: float = 0.02,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
            preserve_order=True,
        )
    _, active = _command_activity(env, command_name, command_threshold)
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = _selected_joint_names(asset, asset_cfg)
    if not joint_names:
        return torch.zeros(env.num_envs, device=env.device)
    name_to_index = {name: i for i, name in enumerate(joint_names)}
    q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]

    pairs = (
        ("FL_hip_joint", "RR_hip_joint", -1.0),
        ("FR_hip_joint", "RL_hip_joint", -1.0),
        ("FL_thigh_joint", "RR_thigh_joint", 1.0),
        ("FR_thigh_joint", "RL_thigh_joint", 1.0),
        ("FL_calf_joint", "RR_calf_joint", 1.0),
        ("FR_calf_joint", "RL_calf_joint", 1.0),
    )
    costs = []
    for a_name, b_name, sign in pairs:
        if a_name in name_to_index and b_name in name_to_index:
            a_idx = name_to_index[a_name]
            b_idx = name_to_index[b_name]
            pos_cost = torch.square(q[:, a_idx] - sign * q[:, b_idx])
            vel_cost = torch.square(qd[:, a_idx] - sign * qd[:, b_idx])
            costs.append(pos_cost + velocity_weight * vel_cost)
    if not costs:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.mean(torch.stack(costs, dim=1), dim=1) * active.float()


def diagonal_foot_symmetry(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.03,
    height_scale: float = 0.06,
    velocity_scale: float = 0.35,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, active = _command_activity(env, command_name, command_threshold)
    _, foot_pos, foot_vel = _foot_body_state(env, asset_cfg)
    if foot_pos.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    vel_xy = torch.norm(foot_vel[:, :, :2], dim=2)
    height_cost = (
        torch.square((foot_pos[:, 0, 2] - foot_pos[:, 3, 2]) / max(height_scale, 1.0e-6))
        + torch.square((foot_pos[:, 1, 2] - foot_pos[:, 2, 2]) / max(height_scale, 1.0e-6))
    )
    velocity_cost = (
        torch.square((vel_xy[:, 0] - vel_xy[:, 3]) / max(velocity_scale, 1.0e-6))
        + torch.square((vel_xy[:, 1] - vel_xy[:, 2]) / max(velocity_scale, 1.0e-6))
    )
    return 0.5 * (height_cost + 0.25 * velocity_cost) * active.float()


def stance_width(
    env: ManagerBasedRlEnv,
    min_width: float = 0.13,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, foot_pos, _ = _foot_body_state(env, asset_cfg)
    if foot_pos.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    front_width = torch.abs(foot_pos[:, 0, 1] - foot_pos[:, 1, 1])
    rear_width = torch.abs(foot_pos[:, 2, 1] - foot_pos[:, 3, 1])
    return 0.5 * (torch.square(torch.clamp(min_width - front_width, min=0.0)) + torch.square(torch.clamp(min_width - rear_width, min=0.0)))


def terrain_level_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    levels = getattr(terrain, "terrain_levels", None)
    if levels is None:
        return torch.zeros(env.num_envs, device=env.device)
    max_level = torch.clamp(torch.max(levels.float()), min=1.0)
    return levels.float() / max_level
