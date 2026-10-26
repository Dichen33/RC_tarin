from __future__ import annotations

from copy import deepcopy

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.viewer import ViewerConfig
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.envs.mdp import dr as envs_dr

from minidog_native.config.obstacle_profiles import profile_name, selected_families
from minidog_native.config.env_cfgs import (
    flat_env_cfg,
    plum_blossom_env_cfg,
    rough_env_cfg,
)
from minidog_native.mdp.curriculums import terrain_levels_focused_obstacle
from ..mdp.rewards import (
    gap_landing_stability,
    gap_takeoff_progress,
    high_wall_clearance_progress,
    obstacle_feet_air_time,
    obstacle_foot_clearance,
    obstacle_scaled_flat_orientation,
    obstacle_scaled_forward_motion,
    obstacle_scaled_forward_stall,
    obstacle_scaled_term,
    obstacle_scaled_track_lin_vel,
    obstacle_scaled_upright,
    plum_airborne_penalty,
    plum_edge_hit,
    plum_gait_plan,
    plum_lateral_center_hold,
    plum_landing_softness,
    plum_next_foothold,
    plum_overhang_penalty,
    plum_support_pair,
    plum_wheel_on_stone,
    wheel_lock_on_obstacle,
    wheel_mode_leg_activity_l2,
    wheel_velocity_tracking,
    wheels_too_near,
)
from ..mdp.observations import terrain_context
from ..mdp.curriculums import terrain_levels_plum_obstacle_release
from ..mdp.terminations import fell_off_course
from mjlab.managers.termination_manager import TerminationTermCfg
from ..robot_cfg import (
    GO2W_FOOT_BODY_NAMES,
    GO2W_JOINT_NAMES,
    GO2W_LEG_BODY_NAMES,
    GO2W_LEG_JOINT_NAMES,
    GO2W_WHEEL_GEOM_NAMES,
    GO2W_WHEEL_JOINT_NAMES,
    LEG_ACTION_SCALE,
    STAND_HEIGHT,
    WHEEL_ACTION_SCALE,
    WHEEL_RADIUS,
    get_robot_cfg,
)

def _foot_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=GO2W_FOOT_BODY_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=3,
        track_air_time=True,
    )


def _body_sensor(name: str, body_names: tuple[str, ...], history_length: int = 4) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="body", pattern=body_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=history_length,
    )


def _sensors() -> tuple[ContactSensorCfg, ...]:
    return (
        _foot_sensor(),
        _body_sensor("base_ground_contact", ("base",)),
        _body_sensor("self_collision", ("base",)),
        _body_sensor("thigh_ground_touch", ("FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh")),
        _body_sensor("shank_ground_touch", ("FR_calf", "FL_calf", "RR_calf", "RL_calf")),
        _body_sensor("trunk_ground_touch", ("base",)),
        RayCastSensorCfg(
            name="height_scanner",
            frame=ObjRef(type="body", name="base", entity="robot"),
            pattern=GridPatternCfg(resolution=0.08, size=(2.4, 1.0)),
            ray_alignment="yaw",
            max_distance=3.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            debug_vis=False,
        ),
    )


def _actions() -> dict:
    actions = {}
    for joint_name in GO2W_LEG_JOINT_NAMES:
        actions[f"position_{joint_name}"] = JointPositionActionCfg(
            entity_name="robot", actuator_names=(joint_name,), scale=LEG_ACTION_SCALE, use_default_offset=True
        )
    for joint_name in GO2W_WHEEL_JOINT_NAMES:
        actions[f"velocity_{joint_name}"] = JointVelocityActionCfg(
            entity_name="robot", actuator_names=(joint_name,), scale=WHEEL_ACTION_SCALE, use_default_offset=True
        )
    return actions


# When a run trains exactly one obstacle family, its sparse event rewards must be
# comparable to the dense locomotion reward.  Measured on a 3000-iteration
# plum-only run: course_completion contributed 0.04% of the episode return while
# track_lin_vel contributed 41%, so "actually finishing the obstacle" was worth
# nothing to the optimiser.
#
# At the measured standing income (1.31/step on the stones, task events worth
# 0.004) the sparse events were ~330x too small to change the policy, so "stay
# upright on the entry platform" was the optimum.  These targets are absolute and
# only ever raised, never lowered, so a task that already configured a larger
# value keeps it.
#
# 2026-10-25, after plum_pilot4.  With the earlier 40/12 targets the measured
# per-iteration income was:
#   commanded_forward_motion +0.3187   (65% of all positive income)
#   upright +0.0962, wheels_on_stone +0.0685, landing +0.0466
#   plum_foothold +0.0318, course_completion +0.0175, next_foothold +0.0028
# i.e. creeping forward paid 6.5x more than placing a wheel on the next stone.
# The policy did exactly that: it reached 0.675 of the row-3 course (1.82 m of
# 2.7 m) and never finished (s_l3 = 0.001).  Placement has to dominate, so the
# two event rewards go up 8x and the creeping income on obstacle rows is cut
# (see ``commanded_forward_motion``'s obstacle_scale) to move the budget from
# "keep moving" to "aim a wheel".
#   course_completion 320 -> 0.0175 * 8 = 0.14/iter, the dominant single event
#   plum_foothold     100 -> 0.0318 * 8.3 = 0.265/iter, one stone now pays more
#                            than 500 steps of creeping instead of 58
#   gap_crossing       12 -> unchanged (gap family not tuned this round)
FOCUS_REWARD_TARGETS = {
    "course_completion": 320.0,
    "gap_crossing": 12.0,
    "plum_foothold": 100.0,
}


def _apply_obstacle_reward_balance(cfg) -> None:
    """Damp flat-ground locomotion on obstacle rows and add foothold shaping.

    Measured on a 2000-iteration plum run: ``track_lin_vel`` alone was ~41% of the
    episode return, while stopping to plant a wheel cost ~3.45 per step against a
    one-off +2.5 per stone.  The robot therefore kept rolling and sliding off the
    cubes.  These wrappers scale the locomotion terms down exactly on the
    obstacle rows, mute wheel spin there, and add the foothold terms the
    reference projects use.
    """
    foot_asset = SceneEntityCfg("robot", body_names=GO2W_FOOT_BODY_NAMES, preserve_order=True)
    wheel_asset = SceneEntityCfg("robot", joint_names=GO2W_WHEEL_JOINT_NAMES, preserve_order=True)
    cfg.rewards["track_lin_vel"] = RewardTermCfg(
        func=obstacle_scaled_track_lin_vel,
        weight=cfg.rewards["track_lin_vel"].weight,
        # Near-muted on the stones (0.05): planting a wheel takes ~0.5 s during
        # which the robot cannot hold the commanded velocity, so even 0.2 made
        # "keep rolling" pay more than "step".  Forward motion still pays through
        # ``commanded_forward_motion`` below, which cannot be farmed while parked.
        params={"command_name": "twist", "std": 0.20, "obstacle_scale": 0.05},
    )
    # Forward motion is the one forward-facing payment that cannot be farmed by
    # standing still (it needs actual body velocity), so it is the right place to
    # buy exploration instead of ``course_progress``, which pays absolute progress
    # and therefore also pays a parked robot.
    #
    # 2026-10-25: 1.0 -> 0.4.  Raising it to 1.0 was correct while "the robot does
    # not dare to move" was the bottleneck; plum_pilot4 shows that is solved (rows
    # 0-2 succeed 0.99) and this term had become 65% of all positive income, paying
    # the policy to creep forward without ever aiming a wheel.  The budget now goes
    # to ``plum_foothold`` / ``course_completion`` / ``plum_next_foothold``.
    cfg.rewards["commanded_forward_motion"] = RewardTermCfg(
        func=obstacle_scaled_forward_motion,
        weight=cfg.rewards["commanded_forward_motion"].weight,
        params={
            "command_name": "twist",
            "command_threshold": 0.05,
            "min_command": 0.08,
            "obstacle_scale": 0.4,
        },
    )
    cfg.rewards["commanded_forward_stall"] = RewardTermCfg(
        func=obstacle_scaled_forward_stall,
        weight=cfg.rewards["commanded_forward_stall"].weight,
        params={
            "command_name": "twist",
            "command_threshold": 0.05,
            "min_command": 0.08,
            "obstacle_scale": 0.2,
        },
    )
    if "flat_orientation" in cfg.rewards:
        cfg.rewards["flat_orientation"] = RewardTermCfg(
            func=obstacle_scaled_flat_orientation,
            weight=cfg.rewards["flat_orientation"].weight,
            params={"std": 0.5, "obstacle_scale": 0.2},
        )
    # Level and yaw holding are the two remaining standing-still incomes on the
    # stones: 1.0 (upright) + 1.2 (yaw tracking) per step for staying parked and
    # facing forward.  With velocity tracking damped those become the largest
    # positive terms on the course, so they are damped there too and left at full
    # weight on flat/rough rows.  ``upright`` is a class-based mjlab term and
    # cannot go through ``obstacle_scaled_term``; see obstacle_scaled_upright.
    if "track_yaw_vel" in cfg.rewards:
        yaw_term = cfg.rewards["track_yaw_vel"]
        yaw_term.params = {
            **yaw_term.params,
            "base_func": yaw_term.func,
            "obstacle_scale": 0.25,
        }
        yaw_term.func = obstacle_scaled_term
    if "upright" in cfg.rewards:
        cfg.rewards["upright"] = RewardTermCfg(
            func=obstacle_scaled_upright,
            weight=cfg.rewards["upright"].weight,
            params={**cfg.rewards["upright"].params, "obstacle_scale": 0.25},
        )
    # No reward for spinning the wheels on the stones: a wheel that slips off a
    # 0.20 m face cannot be used as a foot.
    cfg.rewards["wheel_velocity_tracking"].params["obstacle_scale"] = 0.0
    cfg.rewards["wheel_lock_on_obstacle"] = RewardTermCfg(
        func=wheel_lock_on_obstacle,
        weight=-0.05,
        params={"asset_cfg": wheel_asset},
    )
    # AME_Locomotion's feet_too_near: two wheels inside the same 0.40 m gap wedge
    # the robot, so keep the pair distances above 0.15 m on the course.
    cfg.rewards["wheels_too_near"] = RewardTermCfg(
        func=wheels_too_near,
        weight=-0.15,
        params={"asset_cfg": foot_asset, "min_separation": 0.15},
    )
    # AME_Locomotion's foot_clearance_reward: lift a swinging wheel above the
    # cube faces instead of dragging it into the next cube.  Kept at 0.20: lifting
    # still pays by itself, but at 0.45 the term rewarded holding a wheel in the
    # air (air time rose to ~2 s) instead of placing it.
    cfg.rewards["obstacle_foot_clearance"] = RewardTermCfg(
        func=obstacle_foot_clearance,
        weight=0.20,
        params={"asset_cfg": foot_asset, "target_height": 0.10},
    )
    # "Walk the stones accurately" is repeated placement, but a per-step payment
    # for *being* on a stone is farmable: standing still on the entry platform
    # with one front wheel on the first cube collected this every step, and that
    # local optimum is what the zero-success runs converged to.  Placement is now
    # paid once per newly visited stone by ``plum_foothold``; this counter stays
    # available as a diagnostic (``env._plum_wheels_on_stone``) and can be
    # re-enabled without re-deriving it.
    # Posture and symmetry shaping must not fight single-wheel stepping.  The
    # wrapper receives the term's own function as ``base_func`` instead of
    # looking it up by name, so it can never silently substitute another
    # implementation.  Left/right and diagonal mirror priors are correct on
    # flat ground but oppose the asymmetric stance a single-wheel placement
    # needs, so they are switched off exactly on the obstacle rows.
    import inspect

    for name, scale in (("leg_mirror", 0.0), ("diagonal_leg_symmetry", 0.0)):
        if name not in cfg.rewards:
            continue
        term = cfg.rewards[name]
        if inspect.isclass(term.func):
            continue
        term.params = {
            **term.params,
            "base_func": term.func,
            "obstacle_scale": scale,
        }
        term.func = obstacle_scaled_term
    # ``pose`` is mjlab's class-based ``variable_posture``: the reward manager
    # instantiates it, so it cannot be wrapped by ``obstacle_scaled_term``.  At
    # +0.8/step it paid more for holding the default stance than the sparse
    # course events paid for crossing, which is the main reason the policy
    # parked in front of the first stone.  Lower it here; the remaining
    # height/orientation terms still keep the base posture sane.
    if "pose" in cfg.rewards:
        cfg.rewards["pose"].weight = 0.10
    # Dense income while wheels are properly planted on cubes.  Kept small (0.10):
    # at 0.50 a wheel resting on the first cube while the robot stood on the entry
    # platform was itself a stable local optimum, so the term paid for not
    # attempting the next gap.  Placement is now paid mostly by ``plum_foothold``
    # (one-off, 12 per new stone); this counter stays as light shaping.
    cfg.rewards["plum_wheels_on_stone"] = RewardTermCfg(
        func=plum_wheel_on_stone,
        weight=0.10,
        params={},
    )
    cfg.rewards["plum_edge_hit"] = RewardTermCfg(
        func=plum_edge_hit,
        weight=-0.50,
        params={},
    )
    # BeamDojo eq. (2): penalize wheel-sole sample points that hang off a cube.
    cfg.rewards["plum_overhang"] = RewardTermCfg(
        func=plum_overhang_penalty,
        weight=-0.20,
        params={},
    )
    # Stepping, not hopping: fewer than two wheels in contact means a flight
    # phase, which cannot be aimed at a 0.20 m cube.
    cfg.rewards["plum_airborne"] = RewardTermCfg(
        func=plum_airborne_penalty,
        weight=-1.00,
        params={"sensor_name": "feet_ground_contact", "min_contacts": 2},
    )
    # Straddle the centreline: the two cube rows leave a 0.20 m gutter and the
    # wheels are 0.142 m off-centre, i.e. only 16 mm of lateral margin.  The gate
    # keeps the term off the entry platform, where it otherwise paid the policy
    # for standing still before ever touching a stone.
    cfg.rewards["plum_lateral_hold"] = RewardTermCfg(
        func=plum_lateral_center_hold,
        weight=0.10,
        params={"scale_m": 0.02, "min_body_x": 0.45},
    )
    # Soft gait-structure constraint: the reference gait is a diagonal alternation
    # (the only pair that can be planted together).  Expressed as a penalty for
    # *lacking* any complete diagonal, not as a bonus for having one, so a static
    # four-wheel stance (which has both diagonals) pays nothing.  Disabled for
    # the first clean baseline: a four-wheel stance is free, so the term never
    # pushed the policy to lift a wheel, while penalizing exactly the moment a
    # wheel leaves the ground.  Re-enable once first-mount is reliable.
    cfg.rewards["plum_support_pair"] = RewardTermCfg(
        func=plum_support_pair,
        weight=0.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    # Precomputed reference plan: the *correct* diagonal at the current body
    # position (progress-indexed, so no timing is forced).  Also a mismatch
    # penalty: on a four-wheel stance both diagonals are complete, so the plan is
    # satisfied and a parked robot neither earns nor pays anything for it.
    # Disabled for the same reason as ``plum_support_pair``.
    cfg.rewards["plum_gait_plan"] = RewardTermCfg(
        func=plum_gait_plan,
        weight=0.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    # Reference footstep sequence (arXiv 2403.03639 does this with an MCTS search;
    # our course is fixed, so the sequence is analytic).  Dense guidance for the
    # swinging wheel towards the next cube in its own row.
    #
    # 2026-10-25: 0.5 -> 1.5.  This is the only term that pays for *approaching* a
    # stone, i.e. the only term that needs the radar before contact.  In
    # plum_pilot4 it contributed 0.0028 against 0.3187 of "creep forward", so the
    # optimiser had no reason to look ahead.  It is potential-based (the episode
    # sum telescopes to the potential difference), so raising it cannot be farmed
    # by hovering or by swinging in place.
    cfg.rewards["plum_next_foothold"] = RewardTermCfg(
        func=plum_next_foothold,
        weight=1.50,
        params={"asset_cfg": foot_asset, "sigma": 0.25},
    )
    # Landing absorption (patent CN117008592A: touch -> bend legs -> balance).
    cfg.rewards["plum_landing"] = RewardTermCfg(
        func=plum_landing_softness,
        weight=0.30,
        params={"asset_cfg": foot_asset, "vz_scale": 0.15},
    )
    # 2.7 m at the observed 0.114 m/s needs 23.7 s, i.e. more than the 20 s
    # episode gave: the plum success criterion was unreachable by construction.
    cfg.episode_length_s = 30.0


def _apply_focus_rewards(cfg) -> None:
    """Raise the obstacle event rewards when only one family is trained."""
    if len(selected_families()) != 1:
        return
    for name, target in FOCUS_REWARD_TARGETS.items():
        if name in cfg.rewards:
            cfg.rewards[name].weight = max(cfg.rewards[name].weight, target)


def _adapt(cfg, obstacle_mode: bool = False):
    joint_asset = SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)
    leg_asset = SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINT_NAMES, preserve_order=True)
    wheel_asset = SceneEntityCfg("robot", joint_names=GO2W_WHEEL_JOINT_NAMES, preserve_order=True)
    # MJLab's actuator DR terms index the configured actuator *groups* (the
    # robot has one position group for legs and one velocity group for wheels),
    # rather than individual MuJoCo actuator names.
    leg_actuator = SceneEntityCfg("robot", actuator_ids=[0])
    wheel_geom_asset = SceneEntityCfg("robot", geom_names=GO2W_WHEEL_GEOM_NAMES, preserve_order=True)
    foot_asset = SceneEntityCfg("robot", body_names=GO2W_FOOT_BODY_NAMES, preserve_order=True)
    cfg.scene.entities = {"robot": get_robot_cfg()}
    cfg.scene.sensors = _sensors()
    cfg.actions = _actions()
    cfg.viewer = ViewerConfig(body_name="base", distance=1.8, elevation=-18.0, azimuth=45.0)
    # Wheel traction and actuator response are dominant Go2W sim2sim
    # mismatch sources.  Model-field events run on reset so each episode sees
    # a new plausible hardware instance instead of one fixed instance per env.
    cfg.events["wheel_joint_friction"] = EventTermCfg(
        func=envs_dr.joint_friction,
        mode="reset",
        params={
            "asset_cfg": wheel_asset,
            "ranges": (0.002, 0.020),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["wheel_geom_friction"] = EventTermCfg(
        func=envs_dr.geom_friction,
        mode="reset",
        params={
            "asset_cfg": wheel_geom_asset,
            "ranges": (0.70, 1.30),
            "operation": "scale",
            "axes": [0],
            "shared_random": True,
        },
    )
    cfg.events["leg_joint_friction"] = EventTermCfg(
        func=envs_dr.joint_friction,
        mode="reset",
        params={
            "asset_cfg": leg_asset,
            "ranges": (0.0005, 0.0040),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["leg_joint_armature"] = EventTermCfg(
        func=envs_dr.joint_armature,
        mode="reset",
        params={
            "asset_cfg": leg_asset,
            "ranges": (0.0001, 0.0015),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["wheel_joint_armature"] = EventTermCfg(
        func=envs_dr.joint_armature,
        mode="reset",
        params={
            "asset_cfg": wheel_asset,
            "ranges": (0.0003, 0.0030),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["leg_joint_damping"] = EventTermCfg(
        func=envs_dr.joint_damping,
        mode="reset",
        params={
            "asset_cfg": leg_asset,
            "ranges": (0.005, 0.030),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["wheel_joint_damping"] = EventTermCfg(
        func=envs_dr.joint_damping,
        mode="reset",
        params={
            "asset_cfg": wheel_asset,
            "ranges": (0.005, 0.030),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["leg_pd_gains"] = EventTermCfg(
        func=envs_dr.pd_gains,
        mode="reset",
        params={
            "asset_cfg": leg_actuator,
            "kp_range": (0.90, 1.10),
            "kd_range": (0.80, 1.20),
            "operation": "scale",
        },
    )
    cfg.events["leg_effort_limit"] = EventTermCfg(
        func=envs_dr.effort_limits,
        mode="reset",
        params={
            "asset_cfg": leg_actuator,
            "effort_limit_range": (0.90, 1.10),
            "operation": "scale",
        },
    )
    cfg.events["leg_encoder_bias"] = EventTermCfg(
        func=envs_dr.encoder_bias,
        mode="reset",
        params={"asset_cfg": leg_asset, "bias_range": (-0.008, 0.008)},
    )
    cfg.events["wheel_encoder_bias"] = EventTermCfg(
        func=envs_dr.encoder_bias,
        mode="reset",
        params={"asset_cfg": wheel_asset, "bias_range": (-0.012, 0.012)},
    )
    cfg.events["robot_inertia_scale"] = EventTermCfg(
        func=envs_dr.pseudo_inertia,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
            # pseudo_inertia interprets alpha exponentially; +/-0.05 is
            # approximately a +/-10% mass/inertia scale, not (0.95, 1.05).
            "alpha_range": (-0.05, 0.05),
        },
    )
    if "critic" in cfg.observations:
        # Privileged task context reduces value interference across terrain
        # families without leaking terrain labels into the deployed actor.
        cfg.observations["critic"].terms["terrain_context"] = ObservationTermCfg(
            func=terrain_context,
            clip=(0.0, 1.0),
        )

    for group_name in ("actor_proprio", "critic"):
        group = cfg.observations[group_name]
        if "joint_pos" in group.terms:
            group.terms["joint_pos"].params["asset_cfg"] = joint_asset
            # Feed the same calibrated encoder error to the actor that a real
            # controller would expose.  The reset event above samples it per
            # episode; critic remains on the same observation contract.
            group.terms["joint_pos"].params["biased"] = True
        if "joint_vel" in group.terms:
            group.terms["joint_vel"].params["asset_cfg"] = joint_asset
    # The point-foot gait clock is a leftover from the MiniDog observation
    # contract.  It ticks at 1.8-2.8 Hz with the forward command and carries no
    # information about the course, while a wheeled robot has no trot phase to
    # synchronise with.  At 2 values x 6 history frames it is pure distraction
    # for the stepping task.
    for group in cfg.observations.values():
        group.terms.pop("gait_phase", None)
    cfg.events["reset_joints"].params["asset_cfg"] = joint_asset
    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=("base",))
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=("base",))
    cfg.rewards["base_height"].params["target_height"] = STAND_HEIGHT
    # Measure the base height above the surface the wheels are actually
    # resting on.  With the spawn plane as reference, every stair, wall and
    # stepping-stone climb is scored as "too tall" for its whole duration.
    cfg.rewards["base_height"].params.update(
        support_sensor="feet_ground_contact",
        foot_asset_cfg=foot_asset,
        wheel_radius=WHEEL_RADIUS,
    )
    cfg.rewards["stand_still"].params["target_height"] = STAND_HEIGHT

    for name in ("pose", "stand_pose", "leg_mirror", "diagonal_leg_symmetry"):
        if name in cfg.rewards:
            cfg.rewards[name].params["asset_cfg"] = leg_asset
    for name in ("stand_foot_geometry", "diagonal_foot_symmetry", "stance_width", "foot_clearance", "foot_swing_height", "foot_slip"):
        if name in cfg.rewards:
            cfg.rewards[name].params["asset_cfg"] = foot_asset

    # Wheel rolling is the primary flat/rough locomotion mode. Leg swing
    # rewards and foot-slip penalties would incorrectly penalize four wheels
    # while they are rolling in contact with the floor.
    for name in ("feet_air_time", "feet_gait_consistency", "foot_clearance", "foot_swing_height", "foot_slip", "diagonal_foot_symmetry"):
        if name in cfg.rewards:
            cfg.rewards[name].weight = 0.0
    cfg.rewards["wheel_velocity_tracking"] = RewardTermCfg(
        func=wheel_velocity_tracking,
        weight=0.75,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_WHEEL_JOINT_NAMES, preserve_order=True),
            "wheel_radius": WHEEL_RADIUS,
            "obstacle_scale": 0.25 if obstacle_mode else 1.0,
        },
    )
    cfg.rewards["wheel_mode_leg_activity"] = RewardTermCfg(
        func=wheel_mode_leg_activity_l2,
        weight=-0.04,
        params={
            "command_name": "twist",
            "asset_cfg": leg_asset,
            "position_weight": 0.25,
            "velocity_weight": 1.0,
        },
    )
    cfg.rewards["gap_takeoff_progress"] = RewardTermCfg(
        func=gap_takeoff_progress,
        weight=0.0,
        params={"command_name": "twist", "gap_start_x": 1.15},
    )
    cfg.rewards["gap_landing_stability"] = RewardTermCfg(
        func=gap_landing_stability,
        weight=0.0,
        # The exit edge is derived from the generated gap width at the
        # environment's difficulty, so no fixed landing position is passed.
        params={"command_name": "twist"},
    )
    cfg.rewards["high_wall_clearance_progress"] = RewardTermCfg(
        func=high_wall_clearance_progress,
        weight=0.0,
        params={
            "command_name": "twist",
            "asset_cfg": foot_asset,
            "wall_centers_x": (1.40, 2.95, 4.50),
            # The foot body origin is the wheel axle, so an axle has to sit one
            # wheel radius above the wall top to have actually cleared it.
            "wheel_radius": WHEEL_RADIUS,
        },
    )
    if "plum_foothold" in cfg.rewards:
        # Same axle/contact conversion for the stepping stones: without it the
        # block-top comparison is off by one wheel radius and no stone is ever
        # registered, which kept the plum-blossom success rate at zero.
        cfg.rewards["plum_foothold"].params["wheel_radius"] = WHEEL_RADIUS
        # The six stones are two staggered rows of three.  A crossing follows
        # one complete row; demanding all six forced a serpentine path.
        cfg.rewards["plum_foothold"].params["require_all_stones"] = False
    if obstacle_mode:
        cfg.rewards["gap_takeoff_progress"].weight = 0.20
        cfg.rewards["gap_landing_stability"].weight = 0.20
        cfg.rewards["high_wall_clearance_progress"].weight = 0.20
    if obstacle_mode and "feet_air_time" in cfg.rewards:
        cfg.rewards["feet_air_time"].func = obstacle_feet_air_time
        cfg.rewards["feet_air_time"].weight = 0.25
        cfg.rewards["feet_air_time"].params["obstacle_difficulty_threshold"] = 0.0
    cfg.rewards["track_lin_vel"].weight = 3.5
    if not obstacle_mode and "pose" in cfg.rewards:
        # Flat and rough pretraining should keep the legs near the standing
        # pose while the wheels learn the primary locomotion mode.
        cfg.rewards["pose"].weight = 0.0
    if "stand_contact" in cfg.rewards:
        cfg.rewards["stand_contact"].weight = -0.35
    if "stand_foot_geometry" in cfg.rewards:
        cfg.rewards["stand_foot_geometry"].weight = -0.30
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = 0.0
    cfg.terminations["base_ground_contact"].params["sensor_name"] = "base_ground_contact"
    cfg.terminations["bad_orientation"].params["limit_angle"] = 1.0
    # The competition course stands on a flat floor, so "must not fall off the
    # stones" cannot be enforced by geometry (there is no pit any more): end the
    # episode the moment a wheel reaches the ground.  Without this the robot can
    # drop between the cubes and drive underneath the course to the exit.
    cfg.terminations["fell_off_course"] = TerminationTermCfg(
        func=fell_off_course,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=GO2W_FOOT_BODY_NAMES, preserve_order=True
            ),
            "terrain_name": "rough_plum_blossom",
            # Every course row now has geometry the policy can be scored on: row 0
            # is a continuous 0.60 m slab, rows 1-2 have gaps narrower than the
            # wheel, row 3 is the first real 0.200 m gap.  Falling therefore
            # counts on every row, including the bootcamp one (where the wheels
            # can only reach the floor by leaving the slab).
            "course_start_difficulty": 0.0,
            # Hard termination from the first step.  The old soft window
            # (12k-16.8k steps) was the wrong borrowing from BeamDojo: their soft
            # stage runs on flat ground with virtual footholds, where a miss is
            # recoverable.  Here the cubes are 0.10-0.20 m high and the wheel
            # radius is 0.086 m, so a wheel that drops into a gap cannot climb
            # back; not terminating only creates garbage time.  Measured in
            # plum_step6: no termination -> falling was free -> the policy bounced
            # (mean air time 0.09 -> 2.15 s) and reached the exit 0% of the time.
            "soft_until_step": 0,
            "soft_until_success": 0.0,
            "soft_max_step": 0,
            # A wheel that momentarily dips into a gap is not instantly fatal: the
            # robot has 1.2 s to climb back onto the cubes.  The slot between the
            # platform edge and the first cube is 0.20 m on row 0, which takes
            # 1.0 s to cross at the plum command floor of 0.20 m/s -- the old
            # 0.5 s window (25 steps) turned every crossing into a guaranteed
            # termination (plum_bootcamp2: plum_progress 0.11, success 0.00).
            "recovery_steps": 60,
        },
    )
    return cfg


def go2w_flat_env_cfg(play: bool = False):
    return _adapt(flat_env_cfg(play=play))


def go2w_rough_env_cfg(play: bool = False):
    return _adapt(rough_env_cfg(play=play))


def go2w_step_env_cfg(play: bool = False):
    """Low-obstacle release stage between wheel pretraining and hard courses."""
    cfg = _adapt(rough_env_cfg(play=play), obstacle_mode=True)
    cfg.rewards["wheel_velocity_tracking"].weight = 0.65
    cfg.rewards["track_lin_vel"].weight = 3.1
    if "pose" in cfg.rewards:
        cfg.rewards["pose"].weight = 0.30
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = -0.80
    if "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].weight = -0.10
    return cfg


def go2w_plum_blossom_env_cfg(play: bool = False):
    cfg = _adapt(plum_blossom_env_cfg(play=play), obstacle_mode=True)
    if not play:
        # Use the Go2W LP-ACRL implementation for the unified task.  The
        # focused gap/wall tasks intentionally keep their interpretable
        # single-obstacle curriculum.
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_plum_obstacle_release,
            params={
                "command_name": "twist",
                "success_distance": 2.7,
                "gap_success_distance": 3.35,
                "fail_distance_ratio": 0.35,
                "wall_success_distance": 4.80,
                "default_success_distance": 4.0,
            },
        )
    # Keep the first Go2W curriculum stage wheel-friendly. Difficulty is
    # released by the existing measured terrain gates, not at iteration zero.
    cfg.rewards["wheel_velocity_tracking"].weight = 0.55
    cfg.rewards["track_lin_vel"].weight = 2.8
    if "feet_air_time" in cfg.rewards:
        cfg.rewards["feet_air_time"].func = obstacle_feet_air_time
        # Rewards lifting a wheel, not flying: the air-time window is capped at
        # 0.20 s (a deliberate step) and the weight is kept below the placement
        # terms so hopping can never out-earn stepping.
        cfg.rewards["feet_air_time"].weight = 0.5
        cfg.rewards["feet_air_time"].params["obstacle_difficulty_threshold"] = 0.0
        cfg.rewards["feet_air_time"].params["threshold_max"] = 0.15
    _apply_obstacle_reward_balance(cfg)
    _apply_focus_rewards(cfg)
    # A zero command makes an environment a pure stand-still episode, and
    # ``stand_still`` charges an unbounded ``joint_deviation_l2`` there: measured
    # -6275 raw per standing episode in plum_bootcamp2, i.e. one outlier term
    # dominated the batch return (Mean reward -13, value loss 180) and taught the
    # policy to park -- the exact behaviour that stalled every crossing at the
    # platform edge.  The plum stage always commands forward motion.
    cfg.commands["twist"].rel_standing_envs = 0.0
    # ``course_progress`` is an absolute-progress payment, not a delta: once the
    # robot reaches the exit platform it keeps collecting 0.08/step for the rest
    # of the 30 s episode, and the same is true of any standstill at its current
    # progress.  The crossing is already paid by ``course_completion`` and
    # ``plum_foothold``, so mute it in the plum task.
    cfg.rewards["course_progress"].weight = 0.0
    # Falling off the course is now hard-terminated from the first step, so the
    # termination penalty is the main cost of a failed attempt.  -15 against a
    # 30 s episode made the policy avoid attempting at all (the measured run
    # parked in front of the first stone); -8 still punishes gambling but leaves
    # a successful crossing (40 + footholds) clearly profitable.  Only the plum
    # task is changed; other tasks keep their -15.
    if "is_terminated" in cfg.rewards:
        cfg.rewards["is_terminated"].weight = -8.0
    return cfg


def _single_obstacle_env_cfg(
    play: bool,
    terrain_name: str,
    success_distance: float,
    history_terrain_names: tuple[str, ...] = (),
    revisit_proportion: float = 0.10,
):
    """Create a focused obstacle task from the shared medium-to-hard course.

    Keeping one terrain column makes success rates interpretable and prevents
    a difficult obstacle from being hidden by easier terrain columns.  The
    terrain generator still owns all difficulty rows, so the same curriculum
    starts below the XML medium anchor and can progress beyond it.
    """
    if terrain_name not in {"flat_gap", "flat_high_wall"}:
        raise ValueError(f"Unsupported focused Go2W terrain: {terrain_name}")

    cfg = _adapt(plum_blossom_env_cfg(play=play), obstacle_mode=True)
    generator = deepcopy(cfg.scene.terrain.terrain_generator)
    if terrain_name not in generator.sub_terrains:
        # The family was switched off through MINIDOG_OBSTACLES; a focused task
        # that cannot find its own obstacle would otherwise fail with a bare
        # KeyError while building the terrain grid.
        raise ValueError(
            f"Focused task needs terrain {terrain_name!r}, which is disabled by "
            f"MINIDOG_OBSTACLES={profile_name()!r}. Enable that family or unset the variable."
        )
    terrain_cfg = deepcopy(generator.sub_terrains[terrain_name])
    history_names = tuple(name for name in history_terrain_names if name in generator.sub_terrains and name != terrain_name)
    history_fraction = min(max(revisit_proportion, 0.0), 0.40)
    if history_names:
        terrain_cfg.proportion = 1.0 - history_fraction
        history_cfgs = {}
        each_fraction = history_fraction / len(history_names)
        for name in history_names:
            history_cfg = deepcopy(generator.sub_terrains[name])
            history_cfg.proportion = each_fraction
            history_cfgs[name] = history_cfg
        generator.sub_terrains = {terrain_name: terrain_cfg, **history_cfgs}
    else:
        terrain_cfg.proportion = 1.0
        generator.sub_terrains = {terrain_name: terrain_cfg}
    generator.num_cols = len(generator.sub_terrains)
    generator.curriculum = not play
    cfg.scene.terrain.terrain_generator = generator

    if not play:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_focused_obstacle,
            params={
                "command_name": "twist",
                "success_distance": success_distance,
                "fail_distance_ratio": 0.35,
                "primary_terrain_name": terrain_name,
            },
        )
    cfg.rewards["course_progress"].params["gap_success_distance"] = success_distance
    cfg.rewards["course_progress"].params["wall_success_distance"] = success_distance
    cfg.rewards["course_completion"].params["gap_success_distance"] = success_distance
    cfg.rewards["course_completion"].params["wall_success_distance"] = success_distance
    # The foothold term is meaningful only on stepping stones, so focused gap
    # and wall tasks must not carry a dead reward into their score.
    cfg.rewards["plum_foothold"].weight = 0.0
    if terrain_name == "flat_gap":
        cfg.rewards["gap_crossing"].weight = 5.0
        cfg.rewards["gap_takeoff_progress"].weight = 0.55
        cfg.rewards["gap_landing_stability"].weight = 0.65
    else:
        cfg.rewards["gap_crossing"].weight = 0.0
        cfg.rewards["high_wall_clearance_progress"].weight = 0.70
    _apply_focus_rewards(cfg)
    return cfg


def go2w_gap_env_cfg(play: bool = False):
    return _single_obstacle_env_cfg(
        play,
        "flat_gap",
        success_distance=3.35,
        history_terrain_names=("flat", "random_rough", "pyramid_stairs"),
        revisit_proportion=0.10,
    )


def go2w_high_wall_env_cfg(play: bool = False):
    # The third wall is at x=6.0 in terrain-local coordinates and the spawn
    # origin is x=1.5; 4.8 m relative displacement clears the final wall
    # while remaining inside the generated 8 m terrain patch.
    return _single_obstacle_env_cfg(
        play,
        "flat_high_wall",
        success_distance=4.80,
        history_terrain_names=("pyramid_stairs", "flat_gap"),
        revisit_proportion=0.10,
    )


def go2w_mixed_env_cfg(play: bool = False):
    """Full adaptive course used only after focused obstacle tasks stabilize."""
    return go2w_plum_blossom_env_cfg(play=play)


