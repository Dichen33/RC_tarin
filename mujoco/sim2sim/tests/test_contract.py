from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from _helpers import ROOT

from dogsim.config import load_config
from dogsim.policies import load_policy
from dogsim.policies.ame import AMEActor

MJLAB_ROOT = Path(r"D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab")


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "configs" / "go2w.yaml")
        cls.policy = load_policy(cls.cfg)

    def test_contract(self) -> None:
        contract = self.policy.contract
        self.assertEqual(contract.architecture, "moe")
        self.assertEqual(contract.action_dim, 16)
        self.assertEqual(contract.proprio_dim, 354)
        self.assertEqual(contract.history_length, 6)
        self.assertEqual(contract.frame_dim, 59)
        self.assertEqual(contract.scan_dim, 403)
        self.assertEqual(contract.scan_size, (2.4, 1.0))

    def test_moe_matches_training_actor(self) -> None:
        """The inference replica must reproduce the training forward exactly."""

        if not (MJLAB_ROOT / "minidog_native" / "models.py").exists():
            self.skipTest("training project not available")
        if str(MJLAB_ROOT) not in sys.path:
            sys.path.insert(0, str(MJLAB_ROOT))
        try:
            from tensordict import TensorDict
            from minidog_native.models import AMEProprioActor
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"training actor unavailable: {exc}")

        checkpoint = torch.load(self.cfg.policy.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint["actor_state_dict"]
        obs = TensorDict(
            {"actor_proprio": torch.zeros(1, 354), "actor_radar": torch.zeros(1, 403)},
            batch_size=[1],
        )
        cnn_cfg = {
            "map_shape": (31, 13),
            "map_feature_dim": 64,
            "num_attention_heads": 4,
            "proprio_history_length": 6,
            "proprio_frame_dim": 59,
            "proprio_term_dims": (3, 3, 3, 2, 16, 16, 16),
            "state_latent_dim": 64,
            "use_return_mask": True,
            "moe_cfg": {
                "enabled": True,
                "top_k": 2,
                "auxiliary_enabled": True,
                "router_temperature": 1.0,
                "router_hidden_dims": (128, 64),
                "expert_hidden_dims": (256, 128),
                "residual_scale": 0.40,
                "action_smoothing": 0.45,
                "action_rate_limit": 0.35,
                "wheel_router_bias": 2.5,
                "router_entropy_coef": 0.002,
                "router_balance_coef": 0.02,
                "router_min_usage_coef": 0.05,
                "router_min_usage": 0.03,
                "router_usage_ema_decay": 0.98,
            },
        }
        distribution_cfg = {
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": 0.4,
            "std_type": "log",
            "std_range": (0.05, 1.0),
        }
        reference = AMEProprioActor(
            obs=obs,
            obs_groups={"actor": ("actor_proprio", "actor_radar")},
            obs_set="actor",
            output_dim=16,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg=distribution_cfg,
            cnn_cfg=cnn_cfg,
        )
        reference.load_state_dict(state, strict=True)
        reference.eval()

        ours = AMEActor(
            action_dim=16,
            proprio_frame_dim=59,
            history_length=6,
            term_dims=(3, 3, 3, 2, 16, 16, 16),
            map_shape=(31, 13),
            use_return_mask=True,
            moe=True,
            residual_scale=0.40,
            action_smoothing=0.45,
            action_rate_limit=0.35,
            action_absolute_limit=4.0,
            router_temperature=1.0,
        )
        ours.load_state_dict(state, strict=True)
        ours.eval()

        rng = np.random.default_rng(7)
        proprio = torch.tensor(rng.normal(0.0, 0.2, (1, 354)).astype(np.float32))
        scan = torch.tensor(rng.uniform(0.0, 1.0, (1, 403)).astype(np.float32))
        with torch.inference_mode():
            expected = reference(
                TensorDict(
                    {"actor_proprio": proprio, "actor_radar": scan},
                    batch_size=[1],
                ),
                stochastic_output=False,
            )
            actual = ours(proprio, scan)
        self.assertLess(float((expected - actual).abs().max()), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
