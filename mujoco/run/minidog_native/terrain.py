from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


def course_geometry_at(cfg, difficulty):
    """``(block_size, block_height, platform_height)`` at a course difficulty.

    Both ranges are ``(easy, hard)``: index 0 is used at difficulty 0 and index 1
    at difficulty 1.  This is the single definition used by the terrain builder,
    the foothold rewards and the offline audit -- they previously each carried
    their own copy, one of them interpolating in the opposite direction, so the
    reward scored placements against a cube size the terrain did not use.

    ``difficulty`` may be a float (builder, audit) or a tensor (rewards, one row
    per environment); the returned values follow the same type, so the reward can
    never drift from the geometry again.

    ``size_ramp_exponent`` shapes the cube-size ramp.  With 1.0 the ramp is linear
    and jumps the same-row wheel gap from 0.133 m to 0.200 m across rows 2-3, i.e.
    straight over the 0.172 m wheel diameter: ``plum_pilot3`` solved row 2 (success
    1.00) and never crossed row 3 in 1000 iterations.  An exponent of 1.28 puts a
    row at 0.170 m -- the transition from "the wheel can still roll through" to
    "the wheel must be lifted" -- while keeping row 0 at 0.600 m and the last row
    at the competition 0.200 m.
    """
    exponent = float(getattr(cfg, "size_ramp_exponent", 1.0))
    if isinstance(difficulty, (int, float)):
        d = float(min(max(float(difficulty), 0.0), 1.0))
    else:
        d = difficulty.clamp(0.0, 1.0)
    shape = d if exponent == 1.0 else d ** exponent
    size_lo, size_hi = cfg.block_size_range
    height_lo, height_hi = cfg.block_height_range
    size = size_lo + shape * (size_hi - size_lo)
    height = height_lo + d * (height_hi - height_lo)
    platform = float(getattr(cfg, "platform_height", 0.2)) * (0.5 + 0.5 * d)
    return size, height, platform


def bridge_row_ys(cfg) -> tuple[float, float]:
    """``(entry_row_y, exit_row_y)`` the wheel-line bridges are attached to.

    A bridge has to cover the row whose cubes leave the widest slot against the
    platform the robot steps off: at the entrance that is the row whose first cube
    is farthest along x, at the exit the row whose last cube stops shortest.  With
    the competition layout this is row B (``+0.2``) at the entry and row A
    (``-0.2``) at the exit.  Derived from ``block_centers`` rather than written as
    literals so a layout change moves the bridge with it.
    """
    first_x: dict[float, float] = {}
    last_x: dict[float, float] = {}
    for x, y in cfg.block_centers:
        row = round(float(y), 6)
        first_x[row] = min(first_x.get(row, float(x)), float(x))
        last_x[row] = max(last_x.get(row, float(x)), float(x))
    if len(first_x) < 2:
        return 0.0, 0.0
    return float(max(first_x, key=first_x.get)), float(min(last_x, key=last_x.get))


def _bridge_edge_offsets(cfg) -> tuple[float, float]:
    """Distance from each platform edge to the nearest cube *centre* of the row
    the bridge covers (entry first, exit last).

    Both are 0.50 m on the centrally symmetric competition layout; the slot a
    bridge has to fill is this minus half the cube size.
    """
    platform_half = float(cfg.platform_width) / 2.0
    entry_y, exit_y = bridge_row_ys(cfg)
    entry_x = min(
        float(x) for x, y in cfg.block_centers if round(float(y), 6) == round(entry_y, 6)
    )
    exit_x = max(
        float(x) for x, y in cfg.block_centers if round(float(y), 6) == round(exit_y, 6)
    )
    return entry_x - platform_half, float(cfg.exit_center_x) - platform_half - exit_x


def _bridge_slot_at(cfg, difficulty: float) -> float:
    """Unsupported wheel-line length between a platform edge and its nearest cube."""
    size, _, _ = course_geometry_at(cfg, difficulty)
    entry_offset, exit_offset = _bridge_edge_offsets(cfg)
    return max(0.0, min(entry_offset, exit_offset) - size / 2.0)


def bridge_length_at(cfg, difficulty: float) -> float:
    """Length of the entry/exit wheel-line bridges on one curriculum row.

    ``_add_course`` builds these boxes and ``plum_foothold`` excludes their
    footprint from the stone-support tests, so both must read this formula.  The
    bridge fades out linearly and is gone at ``bridge_fade_end``, which restores
    the real competition geometry on the hard rows.  Returns 0 for courses that do
    not opt in (``bridge_fade_end <= 0``).
    """
    fade_end = float(getattr(cfg, "bridge_fade_end", 0.0))
    if fade_end <= 0.0:
        return 0.0
    fade = float(np.clip(1.0 - float(difficulty) / fade_end, 0.0, 1.0))
    return _bridge_slot_at(cfg, difficulty) * fade


def bridge_length_tensor(cfg, block_size, difficulty):
    """``bridge_length_at`` on per-env tensors.

    The foothold reward scores every environment against the row it was sampled
    on, so the cube size and difficulty arrive as ``[num_envs]`` fields.  The slot
    is linear in the cube size, which makes the tensor path the same formula with
    the ``min(...)`` folded into a constant.
    """
    fade_end = float(getattr(cfg, "bridge_fade_end", 0.0))
    if fade_end <= 0.0:
        return torch.zeros_like(block_size)
    entry_offset, exit_offset = _bridge_edge_offsets(cfg)
    length = (min(entry_offset, exit_offset) - block_size / 2.0).clamp_min(0.0)
    fade = (1.0 - difficulty / fade_end).clamp(0.0, 1.0)
    return length * fade


@dataclass(kw_only=True)
class PlumBlossomTerrainCfg(SubTerrainCfg):
    """Two staggered rows of three blocks between two raised platforms."""

    # Defaults mirror the plum profile in ``config/course_profiles.py`` (7-row
    # bootcamp ladder, every row carries a course).  They are only used when a task
    # forgets to pass a profile, but a stale default here is exactly the class of
    # bug that silently scored the rewards against a different course (0.12/0.36
    # cubes and a 0.28 m platform were once the live values).
    block_size_range: tuple[float, float] = (0.60, 0.20)
    block_height_range: tuple[float, float] = (0.10, 0.20)
    platform_width: float = 0.80
    platform_height: float = 0.20
    floor_thickness: float = 0.10
    position_jitter: float = 0.0
    height_jitter: float = 0.0
    course_start_difficulty: float = 0.0
    # Competition layout (robocon2027 plum course), relative to the entrance
    # platform centre: 200 mm cubes, first cube 100 mm (row A) / 400 mm (row B)
    # from the platform edge, 200 mm between the two rows, 400 mm clear gap
    # inside a row, exit platform centre at 2.70 m.  Kept identical on every
    # curriculum row so the layout never drifts from the real course; only the
    # cube size/height change with difficulty.
    # Verified against the official ROBOCON plum-blossom figure: platform 800 mm,
    # cubes 200 mm, first cube 100 mm (row A) / 400 mm (row B) from the platform
    # edge, 200 mm between the rows, and **400 mm between any two cubes of the same
    # row**, total length 3500 mm, course centrally symmetric.
    #
    #   row A (first cube 100 mm away): 0.6, 1.2, 1.8   at y = -0.2
    #   row B (first cube 400 mm away): 0.9, 1.5, 2.1   at y = +0.2
    #   gaps from the entry platform: 0.10 / 0.40
    #   gaps inside a row:            0.40 / 0.40 / 0.40
    #   last cube to exit platform:   0.40 (A) / 0.10 (B)   -> symmetric
    #
    # ``robocon2027.v2.xml`` puts the third row-A cube at x = 5.125 (i.e. 1.9).
    # That single value contradicts both the uniform 400 mm pitch and the stated
    # central symmetry (it would leave a 0.50 m gap before it and 0.30 m after),
    # so it is treated as a typo in that file: the spec governs.
    block_centers: tuple[tuple[float, float], ...] = (
        (0.6, -0.2), (1.2, -0.2), (1.8, -0.2),
        (0.9, 0.2), (1.5, 0.2), (2.1, 0.2),
    )
    # Distance from the entrance-platform centre to the exit-platform centre.
    # The foothold reward gates placements to the open span between the two
    # platforms, so this must stay the single definition of where the exit is.
    exit_center_x: float = 2.70
    # The competition layout leaves 100 mm from the entrance platform to row A's
    # first cube but 400 mm to row B's first cube (mirrored at the exit).  Those
    # slots sit directly under a wheel line: the left wheels run at y = +0.142
    # through row B and the right wheels at y = -0.142 through row A.  Without a
    # bridge a wheel steps off the platform into a 0.20-0.30 m slot while the
    # other three wheels still hold the robot, which is why the "rollable"
    # rows 0-2 were not rollable in plum_bootcamp1/2: every episode stalled at
    # the platform edge (plum_progress 0.11, plum_foothold 0.004).  The bridge
    # fades linearly and is gone at ``bridge_fade_end``.
    bridge_fade_end: float = 1.0
    # BeamDojo stage-1 "soft terrain" (arXiv 2502.10363 Sec. IV-C1).  When set, the
    # same-row gaps are physically filled by a strip that is flush with the stone
    # tops and lives in MuJoCo geom group 1, i.e. the ray-cast height scanner
    # (``include_geom_groups=(0,)``) does not see it:
    #   physics      : continuous surface -> a misstep cannot drop the robot
    #   perception   : the real stones -> the foothold reward needs the radar
    # The reward side already scores the *nominal* stone layout analytically, so
    # nothing else has to change: a wheel on the fill is "supported but not on a
    # stone", which is exactly the signal that teaches precise placement.
    soft_fill: bool = False
    # Shape of the cube-size ramp (see ``course_geometry_at``).  1.0 = linear.
    # 1.28 inserts a row whose same-row wheel gap is 0.170 m, i.e. right at the
    # 0.172 m wheel diameter -- the transition from "the wheel can still roll
    # through the slot" to "the wheel must be lifted and placed".  Without that
    # row the ramp jumps 0.133 -> 0.200 m and the policy never learns the step.
    size_ramp_exponent: float = 1.0

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))

        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.size[0] / 2.0, self.size[1] / 2.0, self.floor_thickness / 2.0),
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, -self.floor_thickness / 2.0),
        )
        return self._add_course(
            difficulty=difficulty,
            spec=spec,
            rng=rng,
            base_z=0.0,
            geometries=[TerrainGeometry(geom=floor, color=(0.55, 0.55, 0.55, 1.0))],
        )

    def _add_course(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
        base_z: float,
        geometries: list[TerrainGeometry],
    ) -> TerrainOutput:
        body = spec.body("terrain")

        block_size, block_height, platform_height = course_geometry_at(self, difficulty)

        # Keep the 3.5 m course centered in the terrain patch. The returned
        # origin is the top center of the entrance platform.
        course_x = self.size[0] / 2.0 - 1.75
        course_y = self.size[1] / 2.0
        # Keep the base terrain geometries. The rough-grid subclass passes them
        # in so the course is added on top of rough ground rather than replacing it.

        platform_size = self.platform_width / 2.0
        platform_center_x = course_x + platform_size
        for x in (platform_center_x, platform_center_x + self.exit_center_x):
            platform = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(platform_size, platform_size, platform_height / 2.0),
                pos=(x, course_y, base_z + platform_height / 2.0),
            )
            geometries.append(TerrainGeometry(geom=platform, color=(0.35, 0.62, 0.20, 1.0)))

        # Wheel-line entry/exit bridges (see ``bridge_fade_end``).  The row whose
        # cubes leave the widest slot against each platform gets the bridge, so the
        # wheel line that runs through that row keeps support from the platform
        # onto the first cube.  ``plum_foothold`` reads the same length so a wheel
        # rolling across a bridge is not scored as hanging off a cube.
        bridge_length = bridge_length_at(self, difficulty)
        if bridge_length > 1.0e-4:
            entry_x = platform_size + bridge_length / 2.0
            exit_x = self.exit_center_x - platform_size - bridge_length / 2.0
            entry_y, exit_y = bridge_row_ys(self)
            for bridge_x, bridge_y in ((entry_x, entry_y), (exit_x, exit_y)):
                bridge = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(
                        bridge_length / 2.0,
                        block_size / 2.0,
                        platform_height / 2.0,
                    ),
                    pos=(
                        platform_center_x + bridge_x,
                        course_y + bridge_y,
                        base_z + platform_height / 2.0,
                    ),
                )
                geometries.append(
                    TerrainGeometry(geom=bridge, color=(0.35, 0.62, 0.20, 1.0))
                )

        # Block centres come from the competition layout, expressed relative to
        # the entrance platform centre (which sits at course_x + platform/2).
        centers = tuple(
            (platform_center_x + local_x, local_y)
            for local_x, local_y in self.block_centers
        )
        # ``centers`` is already in patch coordinates; adding ``course_x`` again
        # here used to shift every stone by 2.25 m, past the exit platform, while
        # the rewards still looked for them at 0.6 m from the entry platform.
        for center_x, center_y in centers:
            jitter_x = rng.uniform(-self.position_jitter, self.position_jitter)
            jitter_y = rng.uniform(-self.position_jitter, self.position_jitter)
            height = block_height + rng.uniform(-self.height_jitter, self.height_jitter)
            height = max(0.02, height)
            block = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(block_size / 2.0, block_size / 2.0, height / 2.0),
                pos=(
                    center_x + jitter_x,
                    course_y + center_y + jitter_y,
                    base_z + height / 2.0,
                ),
            )
            geometries.append(TerrainGeometry(geom=block, color=(0.35, 0.62, 0.20, 1.0)))

        # BeamDojo stage-1 soft terrain: one continuous strip per stone row, flush
        # with the stone tops, in geom group 1 so the height scanner cannot see it.
        # It spans the whole open span between the two platforms, which also covers
        # the entry/exit wheel-line slots, so this subsumes the bridges above.
        #
        # Deliberately UNNAMED: every sub-terrain patch is added to the same MjSpec,
        # so a generated name (``fill_{index}``) repeats across patches and MuJoCo
        # fails with "repeated name" -- the same failure mode as the earlier
        # duplicated ``lane_plum_plum``.  Nothing downstream needs the name: the
        # audit filters by geom group, and ``TerrainGeometry`` carries only the geom
        # handle and a colour.
        if bool(getattr(self, "soft_fill", False)):
            row_ys = sorted({round(float(y), 6) for _, y in self.block_centers})
            inner = platform_center_x + platform_size
            outer = platform_center_x + self.exit_center_x - platform_size
            span = max(outer - inner, 1.0e-3)
            # 2 mm below the stone tops: the platform top and the bridge top are both
            # at ``platform_height`` which equals ``block_height`` on every row, so a
            # coplanar fill would create double contacts on the same surface.  The
            # 2 mm step is far below the 55 mm foothold tolerance and keeps the fill
            # strictly inside the course volume.
            fill_height = max(block_height - 0.002, 0.01)
            for row_y in row_ys:
                fill = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(span / 2.0, block_size / 2.0, fill_height / 2.0),
                    pos=(
                        (inner + outer) / 2.0,
                        course_y + row_y,
                        base_z + fill_height / 2.0,
                    ),
                    group=1,
                )
                geometries.append(TerrainGeometry(geom=fill, color=(0.45, 0.45, 0.50, 1.0)))

        origin = np.array((platform_center_x, course_y, base_z + platform_height))
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class RoughPlumBlossomTerrainCfg(PlumBlossomTerrainCfg):
    """Rough ground with a staggered plum-blossom stepping course."""

    rough_grid_width: float = 0.40
    rough_height_range: tuple[float, float] = (0.0, 0.10)
    course_corridor_width: float = 1.10
    course_pit_depth: float = 0.35

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        rough_height = self.rough_height_range[0] + difficulty * (
            self.rough_height_range[1] - self.rough_height_range[0]
        )

        # Generate a shallow rough grid over the complete patch. The stepping
        # course is placed on top of the highest rough-grid level so the blocks
        # remain solidly supported even when the surrounding ground is uneven.
        terrain_height = 1.0
        num_x = max(1, int(self.size[0] / self.rough_grid_width))
        num_y = max(1, int(self.size[1] / self.rough_grid_width))
        cell_x = self.size[0] / num_x
        cell_y = self.size[1] / num_y
        heights = rng.uniform(-rough_height, rough_height, (num_x, num_y))
        geometries: list[TerrainGeometry] = []
        for ix in range(num_x):
            for iy in range(num_y):
                height = float(heights[ix, iy])
                # Once the stepping course is active, remove the lateral
                # bypass. The blocks should be the only usable route.
                if difficulty >= self.course_start_difficulty:
                    cell_center_y = (iy + 0.5) * cell_y
                    if abs(cell_center_y - self.size[1] / 2.0) > self.course_corridor_width / 2.0:
                        height = -self.course_pit_depth
                geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(cell_x / 2.0, cell_y / 2.0, (terrain_height + height) / 2.0),
                    pos=(
                        (ix + 0.5) * cell_x,
                        (iy + 0.5) * cell_y,
                        -terrain_height / 2.0 + height / 2.0,
                    ),
                )
                normalized = 0.5 if rough_height == 0.0 else (height + rough_height) / (2.0 * rough_height)
                shade = float(np.clip(normalized, 0.0, 1.0))
                geometries.append(
                    TerrainGeometry(
                        geom=geom,
                        color=(0.42 + 0.10 * shade, 0.42 + 0.10 * shade, 0.42 + 0.10 * shade, 1.0),
                    )
                )

        # Level 0 is a genuine rough-locomotion warm-up. The obstacle course
        # starts from the next curriculum row, so training progresses from
        # walking to stepping stones and finally to the wall.
        if difficulty < self.course_start_difficulty:
            course_x = self.size[0] / 2.0 - 1.75
            origin = np.array(
                (course_x + self.platform_width / 2.0, self.size[1] / 2.0, rough_height)
            )
            return TerrainOutput(origin=origin, geometries=geometries)

        return self._add_course(
            difficulty=difficulty,
            spec=spec,
            rng=rng,
            base_z=rough_height,
            geometries=geometries,
        )


@dataclass(kw_only=True)
class FlatHighWallTerrainCfg(SubTerrainCfg):
    """Flat ground with three repeated transverse walls and no stepping stones."""

    wall_start_difficulty: float = 0.10
    wall_height_range: tuple[float, float] = (0.08, 0.50)
    wall_length_frac: float = 1.0
    wall_thickness: float = 0.05
    wall_centers_x: tuple[float, float, float] = (2.9, 4.45, 6.0)
    floor_thickness: float = 0.10

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.size[0] / 2.0, self.size[1] / 2.0, self.floor_thickness / 2.0),
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, -self.floor_thickness / 2.0),
        )
        geometries = [TerrainGeometry(geom=floor, color=(0.52, 0.52, 0.52, 1.0))]

        if difficulty >= self.wall_start_difficulty:
            wall_progress = (difficulty - self.wall_start_difficulty) / max(
                1.0 - self.wall_start_difficulty, 1.0e-6
            )
            wall_height = self.wall_height_range[0] + wall_progress * (
                self.wall_height_range[1] - self.wall_height_range[0]
            )
            wall_length = self.wall_length_frac * self.size[1]
            center_y = self.size[1] / 2.0
            for center_x in self.wall_centers_x:
                wall = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(self.wall_thickness / 2.0, wall_length / 2.0, wall_height / 2.0),
                    pos=(center_x, center_y, wall_height / 2.0),
                )
                geometries.append(TerrainGeometry(geom=wall, color=(0.72, 0.30, 0.16, 1.0)))

        # Spawn before the first wall. The exit is after all three walls.
        origin = np.array((1.5, self.size[1] / 2.0, 0.0))
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class FlatGapTerrainCfg(SubTerrainCfg):
    """Flat approach and landing areas separated by a transverse ditch."""

    gap_start_x: float = 2.35
    gap_width_range: tuple[float, float] = (0.10, 0.90)
    gap_depth_range: tuple[float, float] = (0.04, 0.45)
    floor_thickness: float = 0.10
    gap_floor_thickness: float = 0.08

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))

        # Keep the obstacle transverse to the route. Small random variations
        # prevent the policy from memorizing one exact edge and width.
        gap_width = self.gap_width_range[0] + difficulty * (
            self.gap_width_range[1] - self.gap_width_range[0]
        )
        gap_depth = self.gap_depth_range[0] + difficulty * (
            self.gap_depth_range[1] - self.gap_depth_range[0]
        )
        gap_width = max(0.05, gap_width + rng.uniform(-0.015, 0.015))
        gap_depth = max(0.02, gap_depth + rng.uniform(-0.01, 0.01))
        gap_start = self.gap_start_x
        gap_end = min(self.size[0] - 0.8, gap_start + gap_width)
        gap_width = gap_end - gap_start
        center_y = self.size[1] / 2.0
        floor_z = -self.floor_thickness / 2.0

        geometries: list[TerrainGeometry] = []

        def add_floor(x0: float, x1: float, z: float, color: tuple[float, float, float, float]) -> None:
            if x1 <= x0:
                return
            floor = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=((x1 - x0) / 2.0, self.size[1] / 2.0, self.floor_thickness / 2.0),
                pos=((x0 + x1) / 2.0, center_y, z),
            )
            geometries.append(TerrainGeometry(geom=floor, color=color))

        # There is no hidden full-height floor beneath the gap. The lower box
        # is the visible ditch bottom, so the robot can step down but cannot
        # exploit an infinite fall-through hole.
        add_floor(0.0, gap_start, floor_z, (0.52, 0.52, 0.52, 1.0))
        add_floor(gap_end, self.size[0], floor_z, (0.52, 0.52, 0.52, 1.0))

        gap_bottom = -gap_depth - self.gap_floor_thickness / 2.0
        bottom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(gap_width / 2.0, self.size[1] / 2.0, self.gap_floor_thickness / 2.0),
            pos=((gap_start + gap_end) / 2.0, center_y, gap_bottom),
        )
        geometries.append(TerrainGeometry(geom=bottom, color=(0.28, 0.28, 0.28, 1.0)))

        # The origin is before the ditch and the route ends after it. This
        # makes curriculum progress a meaningful measure of crossing success.
        origin = np.array((1.20, center_y, 0.0))
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class LowBarTerrainCfg(SubTerrainCfg):
    """Flat ground with repeated low bars whose clearance decreases by level."""

    clearance_range: tuple[float, float] = (0.24, 0.34)
    bar_radius: float = 0.025
    bar_length_frac: float = 0.85
    num_bars: int = 6
    floor_thickness: float = 0.10

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.size[0] / 2.0, self.size[1] / 2.0, self.floor_thickness / 2.0),
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, -self.floor_thickness / 2.0),
        )
        geometries = [TerrainGeometry(geom=floor, color=(0.50, 0.50, 0.50, 1.0))]

        clearance = self.clearance_range[1] - difficulty * (
            self.clearance_range[1] - self.clearance_range[0]
        )
        bar_z = clearance + self.bar_radius
        bar_length = self.bar_length_frac * self.size[1]
        center_y = self.size[1] / 2.0
        bar_color = (0.18, 0.34, 0.72, 1.0)
        positions = np.linspace(1.0, self.size[0] - 1.0, self.num_bars)

        for center_x in positions:
            bar = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(self.bar_radius, bar_length / 2.0),
                pos=(center_x, center_y, bar_z),
                quat=(np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0, 0.0),
            )
            geometries.append(TerrainGeometry(geom=bar, color=bar_color))
            post_height = bar_z
            for y_offset in (-bar_length / 2.0, bar_length / 2.0):
                post = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=(self.bar_radius, post_height / 2.0),
                    pos=(center_x, center_y + y_offset, post_height / 2.0),
                )
                geometries.append(TerrainGeometry(geom=post, color=bar_color))

        origin = np.array((1.5, center_y, 0.0))
        return TerrainOutput(origin=origin, geometries=geometries)
