from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from sim_config import SimConfig


BASE_TERM_DIMS = (3, 3, 3, 2, 12, 12, 12)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 12):
        super().__init__()
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


class GaussianState(nn.Module):
    """Keep the checkpoint's train-time Gaussian parameter for loading."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))


class AMERadarEncoder(nn.Module):
    """Standalone copy of the training-time AME spatial encoder."""

    def __init__(self, map_shape: tuple[int, int] = (31, 13), feature_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        if feature_dim % heads:
            raise ValueError("AME feature dimension must be divisible by attention heads")
        height, width = map_shape
        forward = torch.linspace(-1.0, 1.0, height).view(1, height, 1).expand(1, height, width)
        lateral = torch.linspace(-1.0, 1.0, width).view(1, 1, width).expand(1, height, width)
        self.register_buffer("coordinates", torch.cat((forward, lateral), dim=0), persistent=False)
        self.map_shape = map_shape
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.token_norm = nn.LayerNorm(feature_dim)
        self.query = nn.Sequential(
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, feature_dim),
        )
        self.attention = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, height_scan: torch.Tensor, state_latent: torch.Tensor) -> torch.Tensor:
        batch_size = height_scan.shape[0]
        height_map = height_scan.reshape(batch_size, 1, *self.map_shape)
        coordinates = self.coordinates.unsqueeze(0).expand(batch_size, -1, -1, -1)
        features = self.convolution(torch.cat((height_map, coordinates), dim=1))
        tokens = self.token_norm(features.flatten(start_dim=2).transpose(1, 2))
        query = self.query(state_latent).unsqueeze(1)
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.output_norm(attended.squeeze(1) + query.squeeze(1))


class AMEActor(nn.Module):
    """Inference-only replica of the configurable AME temporal actor."""

    def __init__(
        self,
        action_dim: int,
        proprio_frame_dim: int,
        history_length: int,
        term_dims: tuple[int, ...],
        map_shape: tuple[int, int] = (31, 13),
    ) -> None:
        super().__init__()
        proprio_dim = proprio_frame_dim * history_length
        self.state_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
        )
        self.terrain_encoder = AMERadarEncoder(map_shape=map_shape)
        self.distribution = GaussianState(action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(64 + 64 + proprio_frame_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.term_dims = term_dims
        self.history_length = history_length

    def _current_proprio(self, history: torch.Tensor) -> torch.Tensor:
        pieces = []
        offset = 0
        for term_dim in self.term_dims:
            term = history[:, offset : offset + term_dim * self.history_length]
            term = term.reshape(history.shape[0], self.history_length, term_dim)
            pieces.append(term[:, -1])
            offset += term_dim * self.history_length
        return torch.cat(pieces, dim=-1)

    def forward(self, proprio: torch.Tensor, height_scan: torch.Tensor) -> torch.Tensor:
        current_proprio = self._current_proprio(proprio)
        state_latent = self.state_encoder(proprio)
        terrain_latent = self.terrain_encoder(height_scan, state_latent)
        return self.mlp(torch.cat((state_latent, terrain_latent, current_proprio), dim=-1))


class AMEGo2WMoEActor(AMEActor):
    """Inference replica of the 16-DoF wheel-first AME residual MoE."""

    def __init__(self, action_dim: int, proprio_frame_dim: int, history_length: int,
                 term_dims: tuple[int, ...], map_shape: tuple[int, int] = (31, 13)):
        super().__init__(action_dim, proprio_frame_dim, history_length, term_dims, map_shape)
        del self.mlp
        fused_dim = 64 + 64 + proprio_frame_dim
        self.wheel_policy = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.router = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 4),
        )
        self.residual_experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(fused_dim, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, action_dim),
            )
            for _ in range(3)
        )
        self.router_temperature = 1.0
        self.residual_scale = 0.35
        self.action_smoothing = 0.45
        self.action_rate_limit = 0.35

    def forward(self, proprio: torch.Tensor, height_scan: torch.Tensor) -> torch.Tensor:
        current_proprio = self._current_proprio(proprio)
        state_latent = self.state_encoder(proprio)
        terrain_latent = self.terrain_encoder(height_scan, state_latent)
        latent = torch.cat((state_latent, terrain_latent, current_proprio), dim=-1)
        wheel_action = self.wheel_policy(latent)
        router_weights = torch.softmax(self.router(latent) / self.router_temperature, dim=-1)
        residuals = torch.stack([expert(latent) for expert in self.residual_experts], dim=1)
        residual = torch.sum(router_weights[:, 1:, None] * residuals, dim=1)
        target = wheel_action + self.residual_scale * torch.tanh(residual)
        last_action = current_proprio[:, -target.shape[-1]:]
        delta = self.action_smoothing * (target - last_action)
        delta = self.action_rate_limit * torch.tanh(delta / self.action_rate_limit)
        return last_action + delta


class PolicyRunner:
    """Load old MLP or new AME checkpoints with training-compatible inputs."""

    def __init__(self, checkpoint: Path, config: SimConfig, device: str = "cpu", scan_size: tuple[float, float] | None = None):
        self.device = torch.device(device)
        checkpoint = checkpoint.expanduser().resolve()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actor_state = state.get("actor_state_dict")
        if actor_state is None:
            raise ValueError(f"Checkpoint has no actor_state_dict: {checkpoint}")

        self.is_moe = "wheel_policy.0.weight" in actor_state and "router.0.weight" in actor_state
        self.is_ame = "state_encoder.0.weight" in actor_state
        first_weight = actor_state.get("wheel_policy.0.weight") if self.is_moe else actor_state.get("mlp.0.weight")
        last_weight = actor_state.get("wheel_policy.6.weight") if self.is_moe else actor_state.get("mlp.6.weight")
        if first_weight is None or last_weight is None:
            raise ValueError("Checkpoint does not contain an MLP, AME, or Go2W MoE actor")
        self.obs_dim = int(first_weight.shape[1])
        self.action_dim = int(last_weight.shape[0])
        self.history_length = config.history_length
        self.action_clip = config.action_clip
        term_dims = tuple(config.observation_term_dims)
        self.term_dims = term_dims
        self.base_dim = sum(term_dims)
        if self.is_moe:
            self.scan_dim = config.scan_dim or 403
            self.obs_dim = self.base_dim * self.history_length + self.scan_dim
            self.single_obs_dim = self.base_dim + self.scan_dim
            self.actor = AMEGo2WMoEActor(
                self.action_dim,
                proprio_frame_dim=self.base_dim,
                history_length=self.history_length,
                term_dims=self.term_dims,
            )
            self.actor.load_state_dict(actor_state, strict=True)
        elif self.is_ame:
            self.scan_dim = config.scan_dim or 403
            self.obs_dim = self.base_dim * self.history_length + self.scan_dim
            self.single_obs_dim = self.base_dim + self.scan_dim
            self.actor = AMEActor(
                self.action_dim,
                proprio_frame_dim=self.base_dim,
                history_length=self.history_length,
                term_dims=self.term_dims,
            )
            self.actor.load_state_dict(actor_state, strict=True)
        else:
            if self.obs_dim % self.history_length:
                raise ValueError(f"Actor observation dimension {self.obs_dim} is not divisible by history length 6")
            self.single_obs_dim = self.obs_dim // self.history_length
            self.scan_dim = self.single_obs_dim - self.base_dim
            if self.scan_dim <= 0:
                raise ValueError(f"Invalid scan dimension {self.scan_dim} in checkpoint")
            self.actor = Actor(self.obs_dim, self.action_dim)
            self.actor.load_state_dict({k: v for k, v in actor_state.items() if k.startswith("mlp.")}, strict=True)
        self.actor.to(self.device).eval()
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.history: list[np.ndarray] = []

        if scan_size is None:
            if config.scan_size is not None:
                scan_size = config.scan_size
            elif self.scan_dim == 216:
                scan_size = (1.4, 0.9)
            elif self.scan_dim == 403:
                scan_size = (2.4, 1.0)
            else:
                raise ValueError(
                    f"No default scan geometry for {self.scan_dim} values; "
                    "pass --scan-size X Y or set scan_size in config.yaml"
                )
        self.scan_size = scan_size
        print(f"[Policy] checkpoint={checkpoint}")
        architecture = "AME+wheel-MoE" if self.is_moe else "AME+temporal" if self.is_ame else "MLP-history"
        print(f"[Policy] architecture={architecture}")
        print(f"[Policy] actor_obs={self.obs_dim}, single_obs={self.single_obs_dim}, scan={self.scan_dim}, scan_size={scan_size}")

    def reset(self) -> None:
        self.last_action.fill(0.0)
        self.history.clear()

    def step(self, single_obs: np.ndarray) -> np.ndarray:
        single_obs = np.asarray(single_obs, dtype=np.float32)
        if single_obs.size != self.single_obs_dim:
            raise ValueError(f"Observation has {single_obs.size} values, expected {self.single_obs_dim}")
        if not self.history:
            self.history = [single_obs.copy() for _ in range(self.history_length)]
        else:
            self.history.append(single_obs.copy())
            self.history = self.history[-self.history_length:]

        history = np.asarray(self.history, dtype=np.float32)
        base = history[:, :self.base_dim]
        scan = history[-1, self.base_dim:]
        if scan.size != self.scan_dim:
            raise ValueError(f"Height scan has {scan.size} values, expected {self.scan_dim}")
        if self.is_ame:
            terms = np.split(base, np.cumsum(self.term_dims)[:-1], axis=1)
            proprio = np.concatenate([term.reshape(-1) for term in terms])
            expected_proprio = self.base_dim * self.history_length
            if proprio.size != expected_proprio:
                raise AssertionError(f"Proprioception history has {proprio.size} values, expected {expected_proprio}")
            with torch.inference_mode():
                output = self.actor(
                    torch.from_numpy(proprio).to(self.device).unsqueeze(0),
                    torch.from_numpy(scan).to(self.device).unsqueeze(0),
                )[0]
        else:
            scan_history = history[:, self.base_dim:]
            flat = np.concatenate([term.reshape(-1) for term in np.split(base, np.cumsum(self.term_dims)[:-1], axis=1)] + [scan_history.reshape(-1)])
            if flat.size != self.obs_dim:
                raise AssertionError(f"Grouped observation has {flat.size} values, expected {self.obs_dim}")
            with torch.inference_mode():
                output = self.actor(torch.from_numpy(flat).to(self.device).unsqueeze(0))[0]
        action = output.detach().cpu().numpy().astype(np.float32)
        action = np.clip(action, -self.action_clip, self.action_clip)
        self.last_action = action.copy()
        return action
