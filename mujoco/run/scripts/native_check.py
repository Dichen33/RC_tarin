from __future__ import annotations

import importlib
import importlib.metadata
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_go2w_actor() -> None:
    from go2w_native.config.rl_cfg import go2w_ppo_runner_cfg
    from minidog_native.models import AMEProprioActor

    cfg = go2w_ppo_runner_cfg()
    observations = TensorDict(
        {
            "actor_proprio": torch.zeros(2, 6 * 57),
            "actor_radar": torch.zeros(2, 31 * 13),
        },
        batch_size=[2],
    )
    actor = AMEProprioActor(
        obs=observations,
        obs_groups=cfg.obs_groups,
        obs_set="actor",
        output_dim=16,
        hidden_dims=cfg.actor.hidden_dims,
        activation=cfg.actor.activation,
        obs_normalization=cfg.actor.obs_normalization,
        distribution_cfg=cfg.actor.distribution_cfg,
        cnn_cfg=cfg.actor.cnn_cfg,
    )
    output = actor(observations)
    if not actor.use_moe or tuple(output.shape) != (2, 16):
        raise RuntimeError(
            f"Go2W MoE actor check failed: use_moe={actor.use_moe}, shape={tuple(output.shape)}"
        )
    diagnostics = actor.get_diagnostics()
    expected_diagnostics = {
        "wheel_expert_weight",
        "step_expert_weight",
        "gap_expert_weight",
        "recovery_expert_weight",
        "residual_norm",
        "action_delta",
        "wheel_usage_ema",
        "step_usage_ema",
        "gap_usage_ema",
        "recovery_usage_ema",
        "radar_valid_fraction",
    }
    if not expected_diagnostics.issubset(diagnostics):
        raise RuntimeError(f"Go2W MoE diagnostics mismatch: {sorted(diagnostics)}")
    moe_cfg = cfg.actor.cnn_cfg.get("moe_cfg", {})
    if moe_cfg.get("top_k") != 2:
        raise RuntimeError(f"Go2W MoE top_k mismatch: {moe_cfg.get('top_k')}")
    print("go2w_actor=top2_wheel_plus_residual, action_shape=(2, 16)")


def _check_go2w_16dof_tasks() -> None:
    from go2w_native.config.env_cfgs import (
        go2w_flat_env_cfg,
        go2w_gap_env_cfg,
        go2w_high_wall_env_cfg,
        go2w_mixed_env_cfg,
        go2w_rough_env_cfg,
        go2w_step_env_cfg,
    )
    from go2w_native.config.rl_cfg import go2w_ppo_runner_cfg
    from go2w_native.robot_cfg import GO2W_JOINT_NAMES, GO2W_LEG_JOINT_NAMES, GO2W_WHEEL_JOINT_NAMES

    expected_actions = tuple(
        [f"position_{name}" for name in GO2W_LEG_JOINT_NAMES]
        + [f"velocity_{name}" for name in GO2W_WHEEL_JOINT_NAMES]
    )
    expected_joint_order = tuple(GO2W_JOINT_NAMES)
    for task_name, cfg_fn in (("flat", go2w_flat_env_cfg), ("rough", go2w_rough_env_cfg)):
        cfg = cfg_fn()
        action_names = tuple(cfg.actions)
        if action_names != expected_actions or len(action_names) != 16:
            raise RuntimeError(f"Go2W {task_name} action contract mismatch: {action_names}")
        joint_cfg = cfg.observations["actor_proprio"].terms["joint_pos"].params["asset_cfg"]
        if tuple(joint_cfg.joint_names) != expected_joint_order:
            raise RuntimeError(
                f"Go2W {task_name} observation joint order mismatch: {joint_cfg.joint_names}"
            )
        radar_params = cfg.observations["actor_radar"].terms["height_scan"].params
        if tuple(cfg.observations["actor_proprio"].terms) != (
            "base_ang_vel", "projected_gravity", "command", "joint_pos", "joint_vel", "actions"
        ) or radar_params.get("sensor_name") != "height_scanner":
            raise RuntimeError(f"Go2W {task_name} observation contract mismatch")
        if "terrain_context" not in cfg.observations["critic"].terms:
            raise RuntimeError(f"Go2W {task_name} critic terrain context is missing")
        wheel_reward = cfg.rewards.get("wheel_velocity_tracking")
        if wheel_reward is None or wheel_reward.weight <= 0.0:
            raise RuntimeError(f"Go2W {task_name} wheel tracking reward is not active")
        air_time = cfg.rewards.get("feet_air_time")
        if air_time is not None and air_time.weight != 0.0:
            raise RuntimeError(f"Go2W {task_name} feet_air_time must be disabled outside obstacle mode")
    for task_name, residual_scale, expected in (
        ("flat", 0.0, False),
        ("rough", 0.0, False),
        ("step", 0.20, True),
    ):
        moe_cfg = go2w_ppo_runner_cfg(task_name, residual_scale=residual_scale).actor.cnn_cfg["moe_cfg"]
        if bool(moe_cfg.get("auxiliary_enabled")) != expected:
            raise RuntimeError(f"Go2W {task_name} MoE auxiliary gating mismatch")
    flat_cfg = go2w_flat_env_cfg()
    flat_events = flat_cfg.events
    expected_dr_events = {
        "wheel_joint_friction",
        "wheel_geom_friction",
        "leg_joint_friction",
        "leg_joint_armature",
        "wheel_joint_armature",
        "leg_joint_damping",
        "wheel_joint_damping",
        "leg_pd_gains",
        "leg_effort_limit",
        "leg_encoder_bias",
        "wheel_encoder_bias",
        "robot_inertia_scale",
    }
    missing_dr_events = expected_dr_events.difference(flat_events)
    if missing_dr_events:
        raise RuntimeError(f"Go2W sim2sim randomization events are missing: {sorted(missing_dr_events)}")
    if any(flat_events[name].mode != "reset" for name in expected_dr_events):
        raise RuntimeError("Go2W physical domain randomization must resample on episode reset")
    wheel_friction = flat_events["wheel_geom_friction"].params
    if wheel_friction.get("axes") != [0] or wheel_friction.get("shared_random") is not True:
        raise RuntimeError("Wheel geom friction must be symmetric tangential randomization")
    inertia_alpha = flat_events["robot_inertia_scale"].params.get("alpha_range")
    if inertia_alpha != (-0.05, 0.05):
        raise RuntimeError(f"Go2W inertia randomization is outside the intended log range: {inertia_alpha}")
    joint_pos_params = flat_cfg.observations["actor_proprio"].terms["joint_pos"].params
    if joint_pos_params.get("biased") is not True:
        raise RuntimeError("Actor joint positions must consume the randomized encoder bias")
    step_cfg = go2w_step_env_cfg()
    step_air_time = step_cfg.rewards.get("feet_air_time")
    if step_air_time is None or step_air_time.weight <= 0.0:
        raise RuntimeError("Go2W step task must enable gated feet_air_time")
    flat_reward_params = go2w_flat_env_cfg().rewards["wheel_velocity_tracking"].params
    obstacle_reward_params = step_cfg.rewards["wheel_velocity_tracking"].params
    if flat_reward_params.get("obstacle_scale") != 1.0:
        raise RuntimeError("Flat Go2W wheel tracking must remain fully active")
    if not (0.0 < obstacle_reward_params.get("obstacle_scale", 1.0) < 1.0):
        raise RuntimeError("Obstacle Go2W wheel tracking must be attenuated")
    # The wheeled course must not inherit the point-foot gait priors.  These
    # values are the reward-side contract for a clean stepping baseline.
    course_cfg = go2w_mixed_env_cfg()
    if course_cfg.rewards["feet_gait_consistency"].weight != 0.0:
        raise RuntimeError("Go2W course must not use the point-foot gait clock")
    if course_cfg.rewards["pose"].weight >= 0.5:
        raise RuntimeError("Go2W course pose reward still dominates the sparse events")
    for name in ("leg_mirror", "diagonal_leg_symmetry"):
        if course_cfg.rewards[name].params.get("obstacle_scale") != 0.0:
            raise RuntimeError(f"Go2W course must disable {name} on obstacle rows")
    for name in ("plum_support_pair", "plum_gait_plan"):
        if course_cfg.rewards[name].weight != 0.0:
            raise RuntimeError(f"Go2W baseline must keep {name} disabled")
    gap_cfg = go2w_gap_env_cfg()
    gap_params = gap_cfg.rewards["gap_takeoff_progress"].params
    gap_landing_params = gap_cfg.rewards["gap_landing_stability"].params
    wall_params = go2w_high_wall_env_cfg().rewards["high_wall_clearance_progress"].params
    if gap_params.get("gap_start_x") != 1.15:
        raise RuntimeError("Go2W gap takeoff is not expressed in spawn-relative coordinates")
    # The landing edge is derived from the generated gap width, not fixed to
    # the widest ditch, so a configured value here would be a stale coordinate.
    if gap_landing_params.get("gap_end_x") is not None:
        raise RuntimeError("Go2W gap landing must derive its exit edge from the terrain")
    if tuple(wall_params.get("wall_centers_x", ())) != (1.40, 2.95, 4.50):
        raise RuntimeError("Go2W wall shaping coordinates are misaligned")
    wall_target = go2w_high_wall_env_cfg().rewards["course_completion"].params.get("wall_success_distance")
    if wall_target != 4.80:
        raise RuntimeError(f"Go2W high-wall target is outside the terrain patch: {wall_target}")
    focused = (
        ("gap", go2w_gap_env_cfg, "flat_gap"),
        ("high_wall", go2w_high_wall_env_cfg, "flat_high_wall"),
        ("mixed", go2w_mixed_env_cfg, "rough_plum_blossom"),
    )
    for task_name, cfg_fn, terrain_name in focused:
        cfg = cfg_fn()
        terrain_generator = cfg.scene.terrain.terrain_generator
        terrain_names = tuple(terrain_generator.sub_terrains)
        if task_name == "gap" and terrain_names != ("flat_gap", "flat", "random_rough", "pyramid_stairs"):
            raise RuntimeError(f"Go2W gap history terrain profile mismatch: {terrain_names}")
        if task_name == "high_wall" and terrain_names != ("flat_high_wall", "pyramid_stairs", "flat_gap"):
            raise RuntimeError(f"Go2W high-wall history terrain profile mismatch: {terrain_names}")
        if task_name == "mixed" and terrain_name not in terrain_names:
            raise RuntimeError(f"Go2W mixed terrain profile mismatch: {terrain_names}")
        if task_name != "mixed" and terrain_generator.sub_terrains[terrain_name].proportion < 0.89:
            raise RuntimeError(f"Go2W {task_name} primary terrain proportion is too low")
        if task_name != "mixed" and terrain_generator.num_cols != len(terrain_names):
            raise RuntimeError(f"Go2W {task_name} terrain profile mismatch")
        if tuple(cfg.actions) != expected_actions or len(cfg.actions) != 16:
            raise RuntimeError(f"Go2W {task_name} action contract mismatch")
        curriculum = cfg.curriculum.get("terrain_levels")
        expected_curriculum = (
            "terrain_levels_plum_obstacle_release"
            if task_name == "mixed"
            else "terrain_levels_focused_obstacle"
        )
        if curriculum is None or getattr(curriculum.func, "__name__", "") != expected_curriculum:
            raise RuntimeError(f"Go2W {task_name} curriculum mismatch")
        if task_name == "mixed" and "go2w_native" not in getattr(curriculum.func, "__module__", ""):
            raise RuntimeError("Go2W mixed task is not using the LP-ACRL curriculum")
        air_time = cfg.rewards.get("feet_air_time")
        if air_time is None or air_time.weight <= 0.0:
            raise RuntimeError(f"Go2W {task_name} must enable gated feet_air_time")
    from go2w_native.config.stage_profiles import STAGE_PROFILES
    if tuple(profile.name for profile in STAGE_PROFILES) != (
        "flat", "rough", "step", "gap", "high_wall", "plum_blossom", "mixed"
    ):
        raise RuntimeError("Go2W stage profile order mismatch")
    if STAGE_PROFILES[3].transfer_mode != "actor_only_reset_critic_optimizer":
        raise RuntimeError("Go2W obstacle stages must reset critic and optimizer")
    print("go2w_tasks=flat_rough_16dof_aligned, focused_gap_wall_mixed=enabled, moe_diagnostics=enabled")


def _check_curriculum_sampling_caps() -> None:
    """Regression test that LP sampling cannot bypass per-family caps."""
    from go2w_native.mdp.curriculums import _sample_learning_progress_tasks

    terrain = SimpleNamespace(
        terrain_origins=torch.zeros(3, 2, 3),
        _lp_probabilities=torch.full((6,), 1.0 / 6.0),
        terrain_types=torch.zeros(4, dtype=torch.long),
        terrain_levels=torch.zeros(4, dtype=torch.long),
        env_origins=torch.zeros(4, 3),
        _type_level_caps=torch.tensor([0, 1], dtype=torch.long),
    )
    env_ids = torch.arange(4, dtype=torch.long)
    _sample_learning_progress_tasks(
        terrain,
        env_ids,
        torch.tensor([0, 1], dtype=torch.long),
        max_level=2,
        level_caps=terrain._type_level_caps,
    )
    caps = terrain._type_level_caps[terrain.terrain_types]
    if torch.any(terrain.terrain_levels > caps):
        raise RuntimeError("LP sampler bypassed a terrain-family difficulty cap")
    print("curriculum_sampling_caps=ok")


def main() -> int:
    print(f"python={sys.executable}")
    try:
        import mjlab  # noqa: F401
    except Exception as exc:
        print("mjlab=missing")
        print(f"error={type(exc).__name__}: {exc}")
        print("hint=run: uv sync --extra native --extra cu128")
        return 1

    try:
        print(f"mjlab={importlib.metadata.version('mjlab')}")
    except importlib.metadata.PackageNotFoundError:
        print("mjlab=importable")

    try:
        importlib.import_module("minidog_native")
        importlib.import_module("go2w_native")
    except Exception as exc:
        print("minidog_native=failed")
        print(f"error={type(exc).__name__}: {exc}")
        return 2

    print("minidog_native=ok")
    print("go2w_native=ok")
    _check_go2w_actor()
    _check_go2w_16dof_tasks()
    _check_curriculum_sampling_caps()
    from mjlab.tasks.registry import list_tasks

    native_tasks = [task for task in list_tasks() if task.startswith("Robot-")]
    print(f"registered_tasks={', '.join(native_tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
