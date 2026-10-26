from __future__ import annotations

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Htdw4438Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4
        num_observations = 45
        num_privileged_obs = None
        num_actions = 12
        episode_length_s = 8.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.235]
        default_joint_angles = {
            "fl_hip_joint": 0.0,
            "fl_thigh_joint": 0.9,
            "fl_calf_joint": -1.5,
            "fr_hip_joint": 0.0,
            "fr_thigh_joint": 0.9,
            "fr_calf_joint": -1.5,
            "rl_hip_joint": 0.0,
            "rl_thigh_joint": 0.9,
            "rl_calf_joint": -1.5,
            "rr_hip_joint": 0.0,
            "rr_thigh_joint": 0.9,
            "rr_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 10.0}
        damping = {"joint": 0.3}
        action_scale = 0.25
        decimation = 4

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 3
        resampling_time = 10.0

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.12, 0.16]
            lin_vel_y = [-0.09, 0.09]
            ang_vel_yaw = [-0.8, 0.8]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.235

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.5
            base_height = -1.0
            orientation = -1.0
            action_rate = -0.01
            joint_torques = -0.0


class Htdw4438CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "htdw_4438_standard"
        max_iterations = 1000
        save_interval = 200
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24


class RobotFlatCfg(Htdw4438Cfg):
    class terrain(Htdw4438Cfg.terrain):
        mesh_type = "plane"


class RobotRoughCfg(Htdw4438Cfg):
    class env(Htdw4438Cfg.env):
        episode_length_s = 12.0

    class terrain(Htdw4438Cfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        num_rows = 5
        num_cols = 7
        terrain_proportions = [0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.15]


class RobotCrawlCfg(Htdw4438Cfg):
    class env(Htdw4438Cfg.env):
        episode_length_s = 12.0

    class rewards(Htdw4438Cfg.rewards):
        base_height_target = 0.175

    class commands(Htdw4438Cfg.commands):
        class ranges(Htdw4438Cfg.commands.ranges):
            lin_vel_x = [0.0, 0.12]
            lin_vel_y = [-0.04, 0.04]
            ang_vel_yaw = [-0.4, 0.4]


class RobotFlatCfgPPO(Htdw4438CfgPPO):
    class runner(Htdw4438CfgPPO.runner):
        experiment_name = "robot_flat"


class RobotRoughCfgPPO(Htdw4438CfgPPO):
    class runner(Htdw4438CfgPPO.runner):
        experiment_name = "robot_rough"
        max_iterations = 1500


class RobotCrawlCfgPPO(Htdw4438CfgPPO):
    class runner(Htdw4438CfgPPO.runner):
        experiment_name = "robot_crawl"
        max_iterations = 1500
