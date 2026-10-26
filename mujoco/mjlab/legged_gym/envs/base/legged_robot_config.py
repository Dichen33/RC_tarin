from __future__ import annotations

from legged_gym.envs.base.base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4
        num_observations = 45
        num_privileged_obs = None
        num_actions = 12
        episode_length_s = 8.0

    class init_state:
        pos = [0.0, 0.0, 0.235]
        default_joint_angles = {}

    class control:
        control_type = "P"
        stiffness = {"joint": 10.0}
        damping = {"joint": 0.3}
        action_scale = 0.25
        decimation = 4

    class commands:
        curriculum = True
        max_curriculum = 1.0
        num_commands = 3
        resampling_time = 10.0

        class ranges:
            lin_vel_x = [-0.12, 0.16]
            lin_vel_y = [-0.09, 0.09]
            ang_vel_yaw = [-0.8, 0.8]

    class rewards:
        base_height_target = 0.235

        class scales:
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.5
            base_height = -1.0
            orientation = -1.0
            action_rate = -0.01
            torques = -0.0


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1

    class policy:
        actor_hidden_dims = [256, 256, 128]
        critic_hidden_dims = [256, 256, 128]
        activation = "elu"

    class algorithm:
        entropy_coef = 0.01
        learning_rate = 1e-3
        num_learning_epochs = 5
        num_mini_batches = 4
        gamma = 0.99
        lam = 0.95

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        experiment_name = "htdw_4438_standard"
        run_name = ""
        max_iterations = 1000
        num_steps_per_env = 24
        save_interval = 200

