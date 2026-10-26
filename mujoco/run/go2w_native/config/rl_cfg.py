from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def go2w_ppo_runner_cfg(
    experiment_name: str = "go2w_plum_blossom",
    residual_scale: float = 0.35,
) -> RslRlOnPolicyRunnerCfg:
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
                "proprio_frame_dim": 57,
                "proprio_term_dims": (3, 3, 3, 16, 16, 16),
                "state_latent_dim": 64,
                "use_return_mask": True,
                "moe_cfg": {
                    "enabled": True,
                    "top_k": 2,
                    "auxiliary_enabled": residual_scale > 0.0,
                    "router_temperature": 1.0,
                    "router_hidden_dims": (128, 64),
                    "expert_hidden_dims": (256, 128),
                    "residual_scale": residual_scale,
                    "action_smoothing": 0.45,
                    "action_rate_limit": 0.35,
                    "wheel_router_bias": 2.5,
                    "router_entropy_coef": 0.002,
                    "router_balance_coef": 0.02,
                    "router_min_usage_coef": 0.05,
                    "router_min_usage": 0.03,
                    "router_usage_ema_decay": 0.98,
                },
            },
            distribution_cfg={
                "class_name": "GaussianDistribution",
                # The unified run logged action std growing 0.99 -> 141 because
                # the entropy bonus has no counterweight and the distribution
                # default range is (1e-6, 1e6).  A std of 141 turned every motor
                # command into noise; std_range now makes that impossible.
                "init_std": 0.4,
                "std_type": "log",
                "std_range": (0.05, 1.0),
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            # The critic regresses onto returns built from metre-scale height
            # scans and small velocity rewards; without normalization the value
            # loss diverged to 1e8 in the unified run.  The actor must stay
            # un-normalized (see AMEProprioActor); the critic is never deployed.
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            class_name="minidog_native.algorithms:MoERegularizedPPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # Reverted to 0.0 after plum_bootcamp1 (300 updates): at 0.005 the
            # entropy term drove mean action std 0.40 -> 0.71, bad_orientation
            # peaked at 25, and mean episode length fell 450 -> 86, so the base
            # gate never opened and the plum course was never sampled.  The
            # 0.40 -> 0.28 decay in plum_fixed1 was convergence, not collapse.
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=2.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            # 0.01 kept the adaptive schedule pinned at its 1e-2 ceiling, which
            # is far too aggressive for a 16-DoF integrator action.
            desired_kl=0.02,
            max_grad_norm=1.0,
        ),
        obs_groups={"actor": ("actor_proprio", "actor_radar"), "critic": ("critic",)},
        experiment_name=experiment_name,
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=15000,
    )

