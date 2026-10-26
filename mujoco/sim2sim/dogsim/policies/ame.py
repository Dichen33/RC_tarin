"""Inference replicas of the training AME actor and the Go2W residual MoE."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import PolicyBackend, PolicyContract


class GaussianState(nn.Module):
    """Train-time Gaussian parameter, kept only so checkpoints load strictly."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))


class AMERadarEncoder(nn.Module):
    """AME spatial encoder, including the optional radar return-validity mask.

    The training encoder takes four channels: height, validity, forward
    coordinate, lateral coordinate.  Older checkpoints have a three-channel
    convolution; the loader reads the checkpoint shape and selects the same
    variant here.
    """

    def __init__(
        self,
        map_shape: tuple[int, int] = (31, 13),
        query_dim: int = 64,
        feature_dim: int = 64,
        heads: int = 4,
        use_return_mask: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim % heads:
            raise ValueError("AME feature dimension must be divisible by attention heads")
        height, width = int(map_shape[0]), int(map_shape[1])
        forward = torch.linspace(-1.0, 1.0, height).view(1, height, 1).expand(1, height, width)
        lateral = torch.linspace(-1.0, 1.0, width).view(1, 1, width).expand(1, height, width)
        self.register_buffer("coordinates", torch.cat((forward, lateral), dim=0), persistent=False)
        self.map_shape = (height, width)
        self.use_return_mask = bool(use_return_mask)
        channels = 4 if self.use_return_mask else 3
        self.convolution = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.token_norm = nn.LayerNorm(feature_dim)
        self.query = nn.Sequential(
            nn.Linear(query_dim, 128),
            nn.ELU(),
            nn.Linear(128, feature_dim),
        )
        self.attention = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, height_scan: torch.Tensor, state_latent: torch.Tensor) -> torch.Tensor:
        if height_scan.ndim != 2:
            raise ValueError(f"Expected flattened height scan [B, N], got {tuple(height_scan.shape)}")
        expected = self.map_shape[0] * self.map_shape[1]
        if height_scan.shape[-1] != expected:
            raise ValueError(f"Expected {expected} height samples, got {height_scan.shape[-1]}")
        batch = height_scan.shape[0]
        height_map = height_scan.reshape(batch, 1, *self.map_shape)
        coordinates = self.coordinates.unsqueeze(0).expand(batch, -1, -1, -1)
        if self.use_return_mask:
            valid = (height_map < 0.999).to(height_map.dtype)
            sensor_input = torch.cat((height_map, valid, coordinates), dim=1)
        else:
            sensor_input = torch.cat((height_map, coordinates), dim=1)
        features = self.convolution(sensor_input)
        tokens = features.flatten(start_dim=2).transpose(1, 2)
        tokens = self.token_norm(tokens)
        query_token = self.query(state_latent).unsqueeze(1)
        attended, _ = self.attention(query_token, tokens, tokens, need_weights=False)
        return self.output_norm(attended.squeeze(1) + query_token.squeeze(1))


class AMEActor(nn.Module):
    """AME actor with optional wheel-first residual MoE."""

    def __init__(
        self,
        action_dim: int,
        proprio_frame_dim: int,
        history_length: int,
        term_dims: tuple[int, ...],
        map_shape: tuple[int, int] = (31, 13),
        use_return_mask: bool = True,
        moe: bool = False,
        residual_scale: float = 0.40,
        action_smoothing: float = 0.45,
        action_rate_limit: float = 0.35,
        action_absolute_limit: float = 4.0,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.proprio_frame_dim = int(proprio_frame_dim)
        self.history_length = int(history_length)
        self.term_dims = tuple(int(v) for v in term_dims)
        self.use_moe = bool(moe)
        self.residual_scale = float(residual_scale)
        self.action_smoothing = float(action_smoothing)
        self.action_rate_limit = float(action_rate_limit)
        self.action_absolute_limit = float(action_absolute_limit)
        self.router_temperature = max(float(router_temperature), 1.0e-3)
        proprio_dim = self.proprio_frame_dim * self.history_length
        self.proprio_dim = proprio_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
        )
        self.terrain_encoder = AMERadarEncoder(
            map_shape=map_shape,
            query_dim=64,
            feature_dim=64,
            heads=4,
            use_return_mask=use_return_mask,
        )
        self.distribution = GaussianState(self.action_dim)
        fused = 64 + 64 + self.proprio_frame_dim
        self.fused_dim = fused
        if self.use_moe:
            self.wheel_policy = nn.Sequential(
                nn.Linear(fused, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, self.action_dim),
            )
            self.router = nn.Sequential(
                nn.Linear(fused, 128), nn.ELU(),
                nn.Linear(128, 64), nn.ELU(),
                nn.Linear(64, 4),
            )
            self.residual_experts = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(fused, 256), nn.ELU(),
                    nn.Linear(256, 128), nn.ELU(),
                    nn.Linear(128, self.action_dim),
                )
                for _ in range(3)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(fused, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, self.action_dim),
            )

    def _current_proprio(self, history: torch.Tensor) -> torch.Tensor:
        pieces = []
        offset = 0
        for term_dim in self.term_dims:
            term = history[:, offset : offset + term_dim * self.history_length]
            term = term.reshape(history.shape[0], self.history_length, term_dim)
            pieces.append(term[:, -1])
            offset += term_dim * self.history_length
        return torch.cat(pieces, dim=-1)

    def _latent(self, proprio: torch.Tensor, height_scan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = self._current_proprio(proprio)
        state_latent = self.state_encoder(proprio)
        terrain_latent = self.terrain_encoder(height_scan, state_latent)
        return torch.cat((state_latent, terrain_latent, current), dim=-1), current

    def forward(self, proprio: torch.Tensor, height_scan: torch.Tensor) -> torch.Tensor:
        latent, current = self._latent(proprio, height_scan)
        if not self.use_moe:
            return self.mlp(latent)

        wheel_action = self.wheel_policy(latent)
        router_logits = self.router(latent) / self.router_temperature
        router_weights = torch.softmax(router_logits, dim=-1)
        residual_index = torch.argmax(router_logits[:, 1:], dim=-1)
        residuals = torch.stack([expert(latent) for expert in self.residual_experts], dim=1)
        selected_residual = residuals.gather(
            1, residual_index[:, None, None].expand(-1, 1, self.action_dim)
        ).squeeze(1)
        selected_weight = router_weights[:, 1:].gather(1, residual_index[:, None]).squeeze(1)
        wheel_weight = router_weights[:, 0]
        blend_norm = (wheel_weight + selected_weight).clamp_min(1.0e-6)
        residual_weight = selected_weight / blend_norm
        target = wheel_action + residual_weight[:, None] * self.residual_scale * torch.tanh(
            selected_residual
        )
        last_action = current[:, -self.action_dim :].clamp(
            -self.action_absolute_limit, self.action_absolute_limit
        )
        if last_action.shape[-1] != target.shape[-1]:
            last_action = torch.zeros_like(target)
        delta = self.action_smoothing * (target - last_action)
        if self.action_rate_limit > 0.0:
            delta = self.action_rate_limit * torch.tanh(delta / self.action_rate_limit)
        return (last_action + delta).clamp(
            -self.action_absolute_limit, self.action_absolute_limit
        )


class AMEPolicy(PolicyBackend):
    def __init__(
        self,
        actor: AMEActor,
        contract: PolicyContract,
        device: str = "cpu",
    ) -> None:
        self.actor = actor.to(device).eval()
        self.device = torch.device(device)
        self.contract = contract

    def reset(self) -> None:
        return None

    def act(
        self,
        proprio_history: np.ndarray,
        height_scan: np.ndarray,
        scan_history: np.ndarray | None = None,
    ) -> np.ndarray:
        del scan_history
        proprio = np.asarray(proprio_history, dtype=np.float32).reshape(1, -1)
        scan = np.asarray(height_scan, dtype=np.float32).reshape(1, -1)
        if proprio.shape[1] != self.contract.proprio_dim:
            raise ValueError(
                f"proprio history has {proprio.shape[1]} values, expected {self.contract.proprio_dim}"
            )
        if scan.shape[1] != self.contract.scan_dim:
            raise ValueError(f"scan has {scan.shape[1]} values, expected {self.contract.scan_dim}")
        with torch.inference_mode():
            output = self.actor(
                torch.from_numpy(proprio).to(self.device),
                torch.from_numpy(scan).to(self.device),
            )[0]
        action = output.detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("Policy produced a non-finite action")
        return action
