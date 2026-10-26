"""Go2W-specific termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from go2w_native.robot_cfg import WHEEL_RADIUS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def fell_off_course(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    terrain_name: str = "rough_plum_blossom",
    course_start_difficulty: float = 0.15,
    wheel_radius: float = WHEEL_RADIUS,
    height_fraction: float = 0.5,
    soft_until_step: int = 0,
    soft_until_success: float = 0.0,
    soft_max_step: int = 0,
    recovery_steps: int = 0,
) -> torch.Tensor:
    """Terminate once a wheel touches the ground beside the stepping stones.

    The competition course is a set of 0.20 m cubes standing on a flat floor, so
    a robot that misses a stone simply lands on the ground and could drive on
    underneath the course.  "Must not fall off" is therefore a rule, not
    geometry: a wheel resting on the ground means the crossing has failed.

    A wheel standing on a stone has its axle ``platform_height + wheel_radius``
    above the patch floor, while a wheel on the floor sits at ``wheel_radius``.
    Comparing against half the platform height separates the two cleanly and is
    insensitive to base bobbing.

    ``soft_until_step`` implements BeamDojo's "soft terrain dynamics" stage: until
    the environment has stepped that many times, falling off is not terminal.
    A single misstep terminating the episode is what starves exploration on
    sparse footholds -- the robot never gets a positive sample -- so the first
    part of training only *penalizes* the fall (the wheel-on-ground term in the
    reward) and lets the episode continue.

    ``soft_until_success`` additionally keeps the soft stage until the family's
    success-rate EMA reaches that value.  A pure step budget is a poor proxy: in
    an 800-iteration run the hard constraint switched on at iteration 500, right
    when the robot had started crossing (2.16 stones per episode).  Falls then
    rose 0.05 -> 0.47 and crossing collapsed -- it became timid instead of
    accurate.  Tying the switch to the measured success rate avoids that.
    """
    asset: Entity = env.scene[asset_cfg.name]
    terrain = getattr(env.scene, "terrain", None)
    zeros = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain is None or terrain.terrain_types is None:
        return zeros
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if terrain_name not in names:
        return zeros
    steps = int(getattr(env, "common_step_counter", 0))
    # ``soft_max_step`` is the escape hatch for the deadlock that cost plum_step6:
    # the hard rule only switched on once the family success rate reached 0.10,
    # but the robot never had to avoid falling, so it bounced in the air (mean air
    # time 0.09 -> 2.15 s), never reached the exit (reached 0.00) and therefore
    # never produced the success that would have unlocked the rule.  The cap makes
    # the soft stage bounded no matter how the policy behaves.
    hard_from_cap = soft_max_step > 0 and steps >= int(soft_max_step)
    if not hard_from_cap:
        if soft_until_step > 0 and steps < int(soft_until_step):
            return zeros
        if soft_until_success > 0.0:
            success_ema = getattr(terrain, "_success_ema", None)
            if success_ema is not None:
                rate = float(success_ema[names.index(terrain_name)])
                if rate < float(soft_until_success):
                    return zeros
    course = terrain.terrain_types == names.index(terrain_name)
    if not bool(course.any()):
        return zeros

    config = terrain.cfg.terrain_generator.sub_terrains[terrain_name]
    num_rows = max(int(terrain.terrain_origins.shape[0]), 2)
    difficulty = terrain.terrain_levels.float() / float(num_rows - 1)
    active = course & (difficulty >= float(course_start_difficulty))

    platform_height = float(getattr(config, "platform_height", 0.2)) * 0.5
    # ``env_origins`` for a course patch is the platform top, which is
    # ``2 * platform_height`` above the patch floor under the 0.5 + 0.5 * d ramp.
    floor_threshold = (
        env.scene.env_origins[:, 2] * float(height_fraction) + wheel_radius
    )
    foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    on_floor = (foot_z <= floor_threshold[:, None] + 1.0e-6).any(dim=1)
    fell = active & on_floor
    if recovery_steps <= 0:
        return fell

    # Recovery window (arXiv 2306.17103 "Learning Agile Locomotion on Risky
    # Terrains" trains an explicit recovery-from-slip skill; the same failure mode
    # dominates stepping stones).  A wheel that momentarily dips into the gap is
    # not yet a failed crossing: only give up once the robot has stayed off the
    # cubes for ``recovery_steps`` consecutive steps.  This also teaches the
    # policy to climb back, which the competition needs anyway.
    counter = getattr(env, "_fell_off_counter", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._fell_off_counter = counter
    counter[:] = torch.where(fell, counter + 1, torch.zeros_like(counter))
    return fell & (counter > int(recovery_steps))
