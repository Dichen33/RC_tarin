from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = rough_ppo_runner_cfg()
    cfg.experiment_name = "robot_flat"
    cfg.max_iterations = 10000
    return cfg


def rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name="minidog_native.models:AMEProprioActor",
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            cnn_cfg={
                "map_shape": (31, 13),
                "map_feature_dim": 64,
                "num_attention_heads": 4,
                "proprio_history_length": 6,
                "proprio_frame_dim": 47,
                "proprio_term_dims": (3, 3, 3, 2, 12, 12, 12),
                "state_latent_dim": 64,
            },
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.002,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=2.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "actor": ("actor_proprio", "actor_radar"),
            "critic": ("critic",),
        },
        experiment_name="robot_rough",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=15000,
    )


def crawl_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = rough_ppo_runner_cfg()
    cfg.experiment_name = "robot_crawl"
    return cfg


def plum_blossom_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = rough_ppo_runner_cfg()
    cfg.experiment_name = "robot_plum_blossom"
    cfg.max_iterations = 10000
    return cfg
