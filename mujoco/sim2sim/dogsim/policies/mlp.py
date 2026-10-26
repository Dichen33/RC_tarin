"""Legacy MLP-history actor.

Old checkpoints flatten six proprioception frames and six height-scan frames
into one vector.  This backend preserves that contract exactly so older
MiniDog/Go2W checkpoints remain reproducible.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import PolicyBackend, PolicyContract


class GaussianState(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))


class MLPActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.distribution = GaussianState(action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class MLPPolicy(PolicyBackend):
    def __init__(
        self,
        actor: MLPActor,
        contract: PolicyContract,
        base_dim: int,
        scan_dim: int,
        action_clip: float = 5.0,
        device: str = "cpu",
    ) -> None:
        self.actor = actor.to(device).eval()
        self.device = torch.device(device)
        self.contract = contract
        self.base_dim = int(base_dim)
        self.scan_dim = int(scan_dim)
        self.action_clip = float(action_clip)

    def reset(self) -> None:
        return None

    def act(
        self,
        proprio_history: np.ndarray,
        height_scan: np.ndarray,
        scan_history: np.ndarray | None = None,
    ) -> np.ndarray:
        if scan_history is None:
            raise ValueError("MLP-history checkpoints need a scan history")
        history = np.asarray(scan_history, dtype=np.float32).reshape(-1)
        flat = np.concatenate((np.asarray(proprio_history, dtype=np.float32), history))
        expected = self.contract.proprio_dim + self.contract.history_length * self.scan_dim
        if flat.size != expected:
            raise ValueError(f"MLP observation has {flat.size} values, expected {expected}")
        with torch.inference_mode():
            output = self.actor(torch.from_numpy(flat).to(self.device).unsqueeze(0))[0]
        action = output.detach().cpu().numpy().astype(np.float32)
        action = np.clip(action, -self.action_clip, self.action_clip)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("Policy produced a non-finite action")
        return action
