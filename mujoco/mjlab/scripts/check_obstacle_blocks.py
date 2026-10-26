"""Per-obstacle self-check ("block quality control").

Each obstacle is an independent building block with four interfaces:

1. **terrain**  - the obstacle exists on every curriculum row it claims to cover
2. **reward**   - the reward thresholds match the geometry that is generated
3. **success**  - the success criterion is reachable on every row (no row can be
   an automatic failure, which is what silently disabled the plum course)
4. **metrics**  - which scalars report this block, so training can be judged

This script validates 1-3 offline (CPU MuJoCo only, no GPU, no training) and
prints 4.  Run it before a training run, or after touching any terrain /
reward / curriculum code::

    uv run --frozen --extra native --extra cu128 python scripts/check_obstacle_blocks.py
    uv run --frozen --extra native --extra cu128 python scripts/check_obstacle_blocks.py --all

Exit code 0 = every checked block passed.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mjlab.terrains import TerrainGeneratorCfg  # noqa: E402

from minidog_native.config.obstacle_profiles import (  # noqa: E402
    BASE_TERRAIN_NAMES,
    OBSTACLE_FAMILIES,
    enabled_obstacle_names,
    profile_name,
    selected_families,
)

# Reference numbers measured by forward kinematics on the real robot model.
WHEEL_LIFT_MAX = 0.212       # max wheel contact height with the base held still
BODY_UNDERSIDE = 0.097       # lowest point of the base below the base origin
STAND_HEIGHT = 0.45
PATCH_SIZE = 8.0
# The first row whose same-row slot is wider than the wheel.  The ladder is shaped
# (``size_ramp_exponent = 1.28``) so row 3 sits right at the wheel diameter
# (0.170 m vs 0.172 m, still rollable) and row 4 is the first real drop.  With a
# linear ramp this was row 3 and the 0.133 -> 0.200 m jump skipped the whole
# "roll -> lift" transition in one step, which is why plum_pilot3 solved row 2
# (success 1.00) and never crossed row 3 in 1000 iterations.
EXPECTED_FIRST_DROP_ROW = 4

OK = 'PASS'
BAD = 'FAIL'
WARN = 'WARN'

problems: list[str] = []
lines: list[str] = []


def log(line: str = '') -> None:
    lines.append(line)
    print(line)


def fail(msg: str) -> None:
    problems.append(msg)
    log(f'    {BAD}: {msg}')


# --------------------------------------------------------------------------- #
def sample_sub_terrain(sub_cfg, difficulty: float):
    """Rebuild one sub-terrain at ``difficulty`` and return its boxes.

    Geom group 1 is the BeamDojo stage-1 "soft fill" (``soft_fill``): a continuous
    strip that makes the physics non-droppable while the ray-cast height scanner --
    which reads group 0 only -- still sees the real stones.  Those strips must NOT
    count as stones or as support here, otherwise every gap would measure as closed
    and the rollability invariants would pass on a course that has no gaps.
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='terrain')
    origin = sub_cfg.function(float(difficulty), spec, np.random.default_rng(0)).origin
    boxes = []
    for geom in body.geoms:
        if int(geom.type) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if int(getattr(geom, 'group', 0)) != 0:
            continue
        pos = np.array(geom.pos, dtype=float)
        size = np.array(geom.size, dtype=float)
        boxes.append((pos - size, pos + size))
    return np.array(origin, dtype=float), boxes


def rows(num_rows: int):
    return [(i, i / max(num_rows - 1, 1)) for i in range(num_rows)]


def wheel_line_gap(origin, boxes, wheel_y: float, x_start: float, x_end: float) -> float:
    """Largest run along x with no walking surface under one wheel line.

    ``boxes`` are patch-space AABBs.  Only boxes whose top is the walking surface
    (flush with the platform top, i.e. ``top - origin_z == 0``) count, so the
    patch floor and the base of the platforms are ignored.  This is the check the
    ladder claims but did not have: a row can be "one continuous slab" inside the
    cube rows and still have a slot under a wheel line between the platform edge
    and the first cube.
    """
    spans = []
    for lo, hi in boxes:
        if not (lo[1] - 1e-9 <= wheel_y <= hi[1] + 1e-9):
            continue
        if abs((hi[2] - origin[2]) - 0.0) > 0.01:
            continue
        a, b = max(float(lo[0]), x_start), min(float(hi[0]), x_end)
        if b - a > 1e-6:
            spans.append((a, b))
    if not spans:
        return x_end - x_start
    spans.sort()
    gap = spans[0][0] - x_start
    reach = spans[0][1]
    for a, b in spans[1:]:
        if a > reach:
            gap = max(gap, a - reach)
        reach = max(reach, b)
    return max(gap, x_end - reach)


# --------------------------------------------------------------------------- #
def check_plum(sub_cfg, env_cfg, num_rows: int) -> None:
    from go2w_native.robot_cfg import WHEEL_OFFSETS
    from minidog_native.mdp.rewards import plum_foothold
    from minidog_native.terrain import course_geometry_at

    params = env_cfg.rewards['plum_foothold'].params
    tol = float(params.get('top_tolerance', 0.055))
    radius = float(params.get('wheel_radius', 0.0))
    start = float(sub_cfg.course_start_difficulty)
    track_half = abs(float(WHEEL_OFFSETS['FL'][1]))

    # The 7-row ladder is the fix for a measured deadlock (cap pinned at row 1 for
    # 1500 iterations because the "easy" row already had a 0.224 m gap, wider than
    # the 0.172 m wheel).  A stale copy of this config would look like a normal
    # PASS on every per-row check, so pin the two numbers that define the ladder.
    if abs(float(sub_cfg.block_size_range[0]) - 0.60) > 1e-6 or abs(
        float(sub_cfg.block_size_range[1]) - 0.20
    ) > 1e-6:
        fail(f'plum: block_size_range={tuple(sub_cfg.block_size_range)} != (0.60, 0.20); '
             'the bootcamp row is not a continuous slab (stale course profile?)')
    if num_rows != 7:
        fail(f'plum: {num_rows} rows != 7; the ladder must have three rollable rows '
             'before the first real 0.200 m gap')

    # Same-row pitch and row separation, relative to the cube size, decide whether
    # a row is one continuous slab (row 0) or a course with real gaps.
    first_row_y = float(sub_cfg.block_centers[0][1])
    row_a_x = sorted(
        float(x) for x, y in sub_cfg.block_centers if abs(float(y) - first_row_y) < 1e-6
    )
    same_row_pitch = min(b - a for a, b in zip(row_a_x, row_a_x[1:]))
    row_separation = max(float(y) for _, y in sub_cfg.block_centers) - min(
        float(y) for _, y in sub_cfg.block_centers
    )
    wheel_diameter = 2.0 * radius if radius > 0.0 else 0.172
    gap_rows = []
    drop_rows = []
    wheel_drop_rows = []

    # --- interface 3: a row without a course must still be scoreable ---------
    src = inspect.getsource(plum_foothold.__call__)
    gated = 'exit_success & plum_mask & active_course' in src
    log(f'  row  diff  platform  stone_top  assumed    size   cent   gap  gutter  '
        f'wheelgap  kind  verdict')
    for row, d in rows(num_rows):
        if d < start:
            mark = BAD if gated else OK
            note = 'no course -> stable exit' if not gated else 'no course BUT exit gated by active_course'
            log(f'  {row:>3} {d:>5.2f}      -          -          -       -      -    {mark}  ({note})')
            if gated:
                fail('plum: rows below course_start_difficulty can never be scored '
                     '(plum_foothold gates exit_candidate on active_course)')
            continue
        origin, boxes = sample_sub_terrain(sub_cfg, d)
        platform, stones, halves, centers = None, [], [], []
        for lo, hi in boxes:
            sx = (hi[0] - lo[0]) / 2
            sy = (hi[1] - lo[1]) / 2
            top = hi[2] - origin[2]
            if abs(sx - sub_cfg.platform_width / 2) < 1e-3:
                platform = top if platform is None else max(platform, top)
            elif 0.02 < sx < 0.34 and abs(top) < 0.01 and abs(sx - sy) < 1e-6:
                # Stone tops are flush with the platform on every row, and the
                # rough-grid background is made of boxes too: the half-size test
                # alone is not enough to identify a stone.  The square test keeps
                # the entry/exit wheel-line bridges (same height, non-square)
                # out of the layout, which is compared against ``block_centers``.
                stones.append(top)
                halves.append(sx)
                centers.append(
                    (
                        float((lo[0] + hi[0]) / 2 - origin[0]),
                        float((lo[1] + hi[1]) / 2 - origin[1]),
                    )
                )
        # Largest unsupported run under each wheel line over the whole course.
        course_span_x = (
            float(origin[0] - sub_cfg.platform_width / 2),
            float(origin[0] + sub_cfg.exit_center_x + sub_cfg.platform_width / 2),
        )
        line_gaps = [
            wheel_line_gap(
                origin,
                boxes,
                float(origin[1] + side * track_half),
                course_span_x[0],
                course_span_x[1],
            )
            for side in (1.0, -1.0)
        ]
        max_line_gap = max(line_gaps)
        if max_line_gap > wheel_diameter:
            wheel_drop_rows.append(row)
        block_h = sub_cfg.block_height_range[0] + d * (sub_cfg.block_height_range[1] - sub_cfg.block_height_range[0])
        platform_h = sub_cfg.platform_height * (0.5 + 0.5 * d)
        assumed = block_h - platform_h          # what plum_foothold computes
        measured = float(np.mean(stones)) if stones else float('nan')
        platform_top = float('nan') if platform is None else float(platform)
        # (easy, hard): index 0 at difficulty 0, index 1 at difficulty 1.
        # Read the ramp from the shared helper, so a shaped ramp
        # (``size_ramp_exponent``) cannot desynchronise the audit from the builder.
        size_expect = float(course_geometry_at(sub_cfg, d)[0]) / 2
        mean_half = float(np.mean(halves)) if halves else float('nan')
        # The foothold reward looks for the stones at ``block_centers`` relative
        # to the entrance-platform origin.  A coordinate regression (the stones
        # were once offset by a duplicated ``course_x``) passes every size and
        # height check but makes the course unscoreable, so verify the layout.
        expected_layout = sorted(
            (round(float(x), 2), round(float(y), 2)) for x, y in sub_cfg.block_centers
        )
        measured_layout = sorted((round(x, 2), round(y, 2)) for x, y in centers)
        layout_ok = len(measured_layout) == len(expected_layout) and all(
            abs(mx - ex) <= 0.02 and abs(my - ey) <= 0.02
            for (mx, my), (ex, ey) in zip(measured_layout, expected_layout)
        )
        height_ok = abs(measured - assumed) < 0.01
        platform_ok = abs(platform_top) < 0.01
        size_ok = abs(mean_half - size_expect) < 0.01
        if not height_ok:
            fail(f'plum row {row}: stone top {measured:.3f} m vs reward assumption {assumed:.3f} m')
        if not platform_ok:
            fail(f'plum row {row}: platform top {platform_top:.3f} m above the origin')
        if not size_ok:
            fail(f'plum row {row}: stone half-size {mean_half:.3f} m vs expected {size_expect:.3f} m')
        if not layout_ok:
            fail(f'plum row {row}: stone layout {measured_layout} != cfg {expected_layout} '
                 f'(the terrain and plum_foothold would disagree on where the stones are)')
        # ``plum_foothold`` certifies progress only on rows with a real gap; this
        # prints the same classification so a "blocks = 6" row is visible.
        cube_size = 2.0 * mean_half
        same_row_gap = same_row_pitch - cube_size
        row_gutter = row_separation - cube_size
        is_step = (same_row_gap > 0.02) or (row_gutter > 0.02)
        if is_step:
            gap_rows.append(row)
        if (same_row_gap > wheel_diameter) or (row_gutter > wheel_diameter):
            drop_rows.append(row)
        good = height_ok and platform_ok and size_ok and layout_ok
        log(f'  {row:>3} {d:>5.2f} {platform_top:>9.3f} {measured:>10.3f} {assumed:>9.3f} '
            f'{mean_half:>6.3f} {len(centers):>5} {same_row_gap:>5.3f} {row_gutter:>6.3f} '
            f'{max_line_gap:>9.3f} {"step" if is_step else "solid":>5}   {OK if good else BAD}')
    if not gap_rows:
        fail('plum: no row has a real gap; the ladder never becomes stepping stones')
    if not drop_rows:
        fail('plum: no row has a gap wider than the wheel; the course is never a real gap')
    elif min(drop_rows) != EXPECTED_FIRST_DROP_ROW:
        fail(f'plum: first row a wheel can drop into is {min(drop_rows)}, expected '
             f'{EXPECTED_FIRST_DROP_ROW} (earlier rows must be rollable bootcamp rows)')
    # The same invariant measured the way the robot experiences it: along each
    # wheel line, including the platform-to-first-cube slots.  Rows below
    # ``EXPECTED_FIRST_DROP_ROW`` must be rollable end to end; that row is the
    # first line a wheel drops through.
    if not wheel_drop_rows:
        fail('plum: no wheel line ever has a hole wider than the wheel; '
             'the entry/exit bridges never fade out')
    elif min(wheel_drop_rows) != EXPECTED_FIRST_DROP_ROW:
        fail(f'plum: first wheel-line hole a wheel can fall into is row '
             f'{min(wheel_drop_rows)}, expected {EXPECTED_FIRST_DROP_ROW} '
             '(earlier rows must be rollable)')
    log(f'    wheel lines at y=+-{track_half:.3f} m, wheel diameter {wheel_diameter:.3f} m; '
        f'first wheel-line drop row {min(wheel_drop_rows) if wheel_drop_rows else "none"}')
    log(f'    same-row pitch {same_row_pitch:.3f} m, row separation {row_separation:.3f} m, '
        f'wheel diameter {wheel_diameter:.3f} m; first step row '
        f'{min(gap_rows) if gap_rows else "none"}, first drop row '
        f'{min(drop_rows) if drop_rows else "none"}')
    log(f'    stone top tolerance = ±{tol:.3f} m, contact offset = wheel radius {radius:.3f} m')
    log(f'    layout (cfg, relative to the entrance-platform origin): {list(sub_cfg.block_centers)}')


# --------------------------------------------------------------------------- #
def check_gap(sub_cfg, env_cfg, num_rows: int) -> None:
    params = env_cfg.rewards['gap_landing_stability'].params
    spawn = float(params.get('spawn_x', 1.20))
    success_distance = float(env_cfg.rewards['course_completion'].params.get('gap_success_distance', 3.35))
    log(f'  row  diff  gap_start  gap_end  width  reward_edge  measured  verdict')
    for row, d in rows(num_rows):
        origin, boxes = sample_sub_terrain(sub_cfg, d)
        floor_top = max(hi[2] for lo, hi in boxes)
        floors = sorted([b for b in boxes if abs(b[1][2] - floor_top) < 1e-6], key=lambda b: b[0][0])
        if len(floors) < 2:
            fail(f'gap row {row}: could not identify the two floors')
            continue
        gap_start = float(floors[0][1][0])
        gap_end = float(floors[-1][0][0])
        width_expect = sub_cfg.gap_width_range[0] + d * (sub_cfg.gap_width_range[1] - sub_cfg.gap_width_range[0])
        reward_edge = min(sub_cfg.size[0] - 0.8, sub_cfg.gap_start_x + width_expect) - spawn
        measured = gap_end - spawn
        good = abs(measured - reward_edge) < 0.05
        if not good:
            fail(f'gap row {row}: exit edge {measured:.3f} m vs reward assumption {reward_edge:.3f} m')
        log(f'  {row:>3} {d:>5.2f} {gap_start:>10.2f} {gap_end:>8.2f} {gap_end - gap_start:>6.3f} '
            f'{reward_edge:>11.3f} {measured:>9.3f}  {OK if good else BAD}')
    reach = spawn + success_distance
    log(f'    success needs {success_distance:.2f} m -> reaches x={reach:.2f} of {PATCH_SIZE:.1f} m patch '
        f'{"(inside)" if reach < PATCH_SIZE else "(OUTSIDE PATCH)"}')
    if reach >= PATCH_SIZE:
        fail('gap: success distance falls outside the terrain patch')


# --------------------------------------------------------------------------- #
def check_wall(sub_cfg, env_cfg, num_rows: int) -> None:
    params = env_cfg.rewards['high_wall_clearance_progress'].params
    success_distance = float(env_cfg.rewards['course_completion'].params.get('wall_success_distance', 4.80))
    spawn = 1.5
    log(f'  row  diff  walls  wall_h  reward_h  need_base  reach  verdict')
    for row, d in rows(num_rows):
        origin, boxes = sample_sub_terrain(sub_cfg, d)
        walls = [hi[2] - lo[2] for lo, hi in boxes if (hi[0] - lo[0]) < 0.2]
        if not walls:
            log(f'  {row:>3} {d:>5.2f} {0:>6} {"-":>7} {"-":>9} {"-":>10} {"flat":>6}  {OK}  (no wall -> 4.8 m drive)')
            continue
        h = float(np.mean(walls))
        start = float(sub_cfg.wall_start_difficulty)
        prog = max(0.0, (d - start) / max(1 - start, 1e-6))
        assumed = sub_cfg.wall_height_range[0] + prog * (sub_cfg.wall_height_range[1] - sub_cfg.wall_height_range[0])
        need_base = h + BODY_UNDERSIDE
        good = abs(h - assumed) < 0.02
        reach = 'yes' if need_base <= STAND_HEIGHT + 0.30 else 'hard'
        if not good:
            fail(f'wall row {row}: height {h:.3f} m vs reward assumption {assumed:.3f} m')
        log(f'  {row:>3} {d:>5.2f} {len(walls):>6} {h:>7.3f} {assumed:>9.3f} {need_base:>10.3f} '
            f'{reach:>6}  {OK if good else BAD}')
    reach = spawn + success_distance
    log(f'    success needs {success_distance:.2f} m -> reaches x={reach:.2f} of {PATCH_SIZE:.1f} m patch '
        f'{"(inside)" if reach < PATCH_SIZE else "(OUTSIDE PATCH)"}')
    if reach >= PATCH_SIZE:
        fail('wall: success distance falls outside the terrain patch')


# --------------------------------------------------------------------------- #
def check_standard(sub_cfg, env_cfg, num_rows: int) -> None:
    success_distance = float(env_cfg.rewards['course_completion'].params.get('default_success_distance', 4.0))
    log(f'  row  diff  boxes  max_step  wheel_lift_ok  verdict')
    for row, d in rows(num_rows):
        origin, boxes = sample_sub_terrain(sub_cfg, d)
        tops = sorted({round(float(hi[2]), 3) for lo, hi in boxes})
        step = float(np.max(np.diff(tops))) if len(tops) > 1 else 0.0
        lift_ok = step <= WHEEL_LIFT_MAX + 1e-6
        verdict = OK if lift_ok else WARN
        log(f'  {row:>3} {d:>5.2f} {len(boxes):>6} {step:>9.3f} {("yes" if lift_ok else "no"):>13}  '
            f'{verdict}  {"" if lift_ok else "(needs jump/momentum)"}')
    log(f'    success needs {success_distance:.2f} m forward, wheel lift limit {WHEEL_LIFT_MAX:.3f} m')


CHECKS = {
    'plum': ('rough_plum_blossom', check_plum),
    'gap': ('flat_gap', check_gap),
    'wall': ('flat_high_wall', check_wall),
    'stairs': ('pyramid_stairs', check_standard),
}

METRICS = {
    'plum': ('plum_blossom_success_rate_ema', 'rough_plum_blossom_level_cap', 'rough_plum_blossom_blocks_completed'),
    'gap': ('gap_success_rate_ema', 'flat_gap_level_cap', 'Episode_Reward/gap_crossing'),
    'wall': ('high_wall_success_rate_ema', 'flat_high_wall_level_cap', 'Episode_Reward/high_wall_clearance_progress'),
    'stairs': ('standard_success_rate_ema', 'pyramid_stairs_level_cap', 'Metrics/air_time_mean'),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='check every family, ignoring MINIDOG_OBSTACLES')
    args = parser.parse_args()

    from minidog_native.config.course_profiles import plum_course_terrain_generator
    from go2w_native.config.env_cfgs import go2w_mixed_env_cfg

    families = tuple(OBSTACLE_FAMILIES) if args.all else selected_families()
    env_cfg = go2w_mixed_env_cfg()
    gen_cfg: TerrainGeneratorCfg = env_cfg.scene.terrain.terrain_generator
    subs = gen_cfg.sub_terrains
    num_rows = int(gen_cfg.num_rows)

    log('=' * 78)
    log(f'obstacle block check   profile={profile_name()}   rows={num_rows}   '
        f'families={list(families) or "none"}')
    log(f'terrain columns: {list(subs)}')
    log('=' * 78)

    for family in families:
        key = CHECKS.get(family)
        if key is None:
            # blocks / slope share the standard success rule but have no dedicated check
            sub_name = OBSTACLE_FAMILIES[family][0]
            sub_cfg = subs.get(sub_name)
            log(f'\n[{family}]  (no dedicated check; verifying terrain presence only)')
            if sub_cfg is None:
                fail(f'{family}: terrain {sub_name!r} missing from the generated course')
            else:
                log(f'  {OK}: {sub_name} present, {num_rows} rows')
            continue
        sub_name, checker = key
        sub_cfg = subs.get(sub_name)
        log(f'\n[{family}] {sub_name}')
        if sub_cfg is None:
            fail(f'{family}: terrain {sub_name!r} missing from the generated course')
            continue
        checker(sub_cfg, env_cfg, num_rows)
        log('    judgement metrics: ' + ', '.join(METRICS.get(family, ())))

    log('')
    log('=' * 78)
    if problems:
        log(f'RESULT: FAIL  ({len(problems)} problem(s))')
        for p in problems:
            log(f'  - {p}')
    else:
        log('RESULT: PASS  (all checked blocks are consistent and scoreable)')
    log('=' * 78)

    report = ROOT / 'obstacle_block_check.txt'
    try:
        report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'\nreport written to {report}')
    except OSError as exc:
        # Read-only checkouts / sandboxes must not turn a passing audit into a crash.
        print(f'\n[note] could not write the report file ({exc}); the output above is complete')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
