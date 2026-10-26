"""Terrain-aware actors used by the MiniDog and Go2W tasks.

The actor keeps the PPO distribution contract used by RSL-RL.  The optional
MoE path is deliberately residual: a wheel-first policy always produces the
base action and obstacle experts only add a bounded correction.  This makes
the transition into and out of an obstacle continuous instead of replacing
the whole action with an ``argmax`` expert.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_nn_activation


# Hard bound on the residual-MoE action, in normalized policy units (before the
# per-joint action scale).  ``_moe_action`` integrates the previous action, so
# without a bound the command can drift arbitrarily far from the range the PD
# controller was tuned for; the unified run ended with action deltas ~16/step.
ACTION_ABSOLUTE_LIMIT = 4.0


def _as_pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a pair of integers")
    pair = (int(value[0]), int(value[1]))
    if pair[0] < 1 or pair[1] < 1:
        raise ValueError(f"{name} must contain positive integers")
    return pair


class AMERadarEncoder(nn.Module):
    """Encode a local height map with convolution and cross-attention.

    The height scan is converted into three channels: height, normalized
    forward coordinate, and normalized lateral coordinate. The proprioceptive
    latent is used as the query so the policy can select terrain features that
    are relevant to its current gait and body state.
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        query_dim: int,
        feature_dim: int,
        num_attention_heads: int,
        activation: str,
        use_return_mask: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim % num_attention_heads != 0:
            raise ValueError("feature_dim must be divisible by num_attention_heads")

        self.map_shape = map_shape
        self.use_return_mask = bool(use_return_mask)
        height, width = map_shape
        forward = torch.linspace(-1.0, 1.0, height).view(1, height, 1).expand(1, height, width)
        lateral = torch.linspace(-1.0, 1.0, width).view(1, 1, width).expand(1, height, width)
        self.register_buffer("coordinates", torch.cat((forward, lateral), dim=0), persistent=False)

        input_channels = 4 if self.use_return_mask else 3
        self.convolution = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            resolve_nn_activation(activation),
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=1, padding=1),
            resolve_nn_activation(activation),
        )
        self.token_norm = nn.LayerNorm(feature_dim)
        self.query = MLP(query_dim, feature_dim, (128,), activation)
        self.attention = nn.MultiheadAttention(feature_dim, num_attention_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, height_scan: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if height_scan.ndim != 2:
            raise ValueError(f"Expected flattened height scan [B, N], got {tuple(height_scan.shape)}")
        expected = self.map_shape[0] * self.map_shape[1]
        if height_scan.shape[-1] != expected:
            raise ValueError(f"Expected {expected} height samples, got {height_scan.shape[-1]}")

        batch_size = height_scan.shape[0]
        height_map = height_scan.reshape(batch_size, 1, *self.map_shape)
        coordinates = self.coordinates.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if self.use_return_mask:
            # ODIN1-style no-return samples are encoded as the clipped maximum
            # by ``safe_height_scan``.  Exposing validity prevents the encoder
            # from confusing a missing return with a real high obstacle.
            valid = (height_map < 0.999).to(height_map.dtype)
            sensor_input = torch.cat((height_map, valid, coordinates), dim=1)
        else:
            sensor_input = torch.cat((height_map, coordinates), dim=1)
        features = self.convolution(sensor_input)
        tokens = features.flatten(start_dim=2).transpose(1, 2)
        tokens = self.token_norm(tokens)

        query_token = self.query(query).unsqueeze(1)
        attended, _ = self.attention(query_token, tokens, tokens, need_weights=False)
        return self.output_norm(attended.squeeze(1) + query_token.squeeze(1))


class AMEProprioActor(nn.Module):
    """AME terrain encoder plus DreamWaQ++-style proprioceptive latent.

    The temporal branch is feed-forward over six frames. This gives PPO a
    stable, history-aware latent without changing RSL-RL's recurrent rollout
    storage. A true recurrent policy can be added later after this contract is
    validated in training and deployment.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if obs_normalization:
            raise ValueError("AMEProprioActor currently expects obs_normalization=False")
        if distribution_cfg is None:
            raise ValueError("AMEProprioActor requires a distribution_cfg for PPO actions")

        options = dict(cnn_cfg or {})
        moe_cfg = dict(options.pop("moe_cfg", {}) or {})
        self.proprio_group = str(options.pop("proprio_group", "actor_proprio"))
        self.radar_group = str(options.pop("radar_group", "actor_radar"))
        if self.proprio_group not in obs or self.radar_group not in obs:
            raise ValueError(
                f"AMEProprioActor needs '{self.proprio_group}' and '{self.radar_group}' observations; "
                f"received {list(obs.keys())}"
            )
        if self.proprio_group not in obs_groups[obs_set] or self.radar_group not in obs_groups[obs_set]:
            raise ValueError(f"Observation set '{obs_set}' must contain both actor observation groups")
        if obs[self.proprio_group].ndim != 2 or obs[self.radar_group].ndim != 2:
            raise ValueError("AMEProprioActor expects flattened 1D proprioception and radar groups")

        self.proprio_history_length = int(options.pop("proprio_history_length", 6))
        self.proprio_frame_dim = int(options.pop("proprio_frame_dim", 47))
        self.proprio_term_dims = tuple(int(value) for value in options.pop(
            "proprio_term_dims", (3, 3, 3, 2, 12, 12, 12)
        ))
        self.state_latent_dim = int(options.pop("state_latent_dim", 64))
        use_return_mask = bool(options.pop("use_return_mask", True))
        map_shape = _as_pair(options.pop("map_shape", (31, 13)), "map_shape")
        feature_dim = int(options.pop("map_feature_dim", 64))
        num_attention_heads = int(options.pop("num_attention_heads", 4))
        radar_dim = int(obs[self.radar_group].shape[-1])
        proprio_dim = int(obs[self.proprio_group].shape[-1])
        expected_proprio_dim = self.proprio_history_length * self.proprio_frame_dim
        if proprio_dim != expected_proprio_dim:
            raise ValueError(
                f"Expected {expected_proprio_dim} proprioception values, got {proprio_dim}. "
                "Check history_length and proprio_frame_dim."
            )
        if sum(self.proprio_term_dims) != self.proprio_frame_dim:
            raise ValueError("proprio_term_dims must sum to proprio_frame_dim")
        if radar_dim != map_shape[0] * map_shape[1]:
            raise ValueError(f"Expected {map_shape[0] * map_shape[1]} radar values, got {radar_dim}")
        if options:
            raise ValueError(f"Unknown AME model options: {sorted(options)}")

        self.obs_groups = list(obs_groups[obs_set])
        self.obs_dim = proprio_dim + radar_dim
        self.obs_normalizer = nn.Identity()
        self.obs_normalization = False
        # Updated on each forward pass for lightweight training diagnostics.
        # Values are detached tensors so they never retain an autograd graph.
        self._last_diagnostics: dict[str, torch.Tensor] = {}
        self._last_router_weights: torch.Tensor | None = None
        self._last_radar_valid_fraction = torch.zeros(())
        self.register_buffer("expert_usage_ema", torch.zeros(4), persistent=False)

        self.state_encoder = MLP(
            proprio_dim,
            self.state_latent_dim,
            (256, 128),
            activation,
        )
        self.terrain_encoder = AMERadarEncoder(
            map_shape=map_shape,
            query_dim=self.state_latent_dim,
            feature_dim=feature_dim,
            num_attention_heads=num_attention_heads,
            activation=activation,
            use_return_mask=use_return_mask,
        )

        distribution_cfg = deepcopy(distribution_cfg)
        dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
        self.distribution = dist_class(output_dim, **distribution_cfg)
        fused_dim = self.state_latent_dim + feature_dim + self.proprio_frame_dim
        self.action_dim = int(output_dim)

        self.use_moe = bool(moe_cfg.pop("enabled", False))
        self.expert_names = ("wheel", "step", "gap", "recovery")
        # Wheel is always retained as the base expert.  ``top_k=2`` therefore
        # means wheel plus one obstacle expert, avoiding three full residual
        # MLP evaluations for every rollout/update sample.
        self.moe_top_k = max(int(moe_cfg.pop("top_k", 2)), 2)
        self.moe_top_k = min(self.moe_top_k, len(self.expert_names))
        self.router_temperature = max(float(moe_cfg.pop("router_temperature", 1.0)), 1.0e-3)
        self.residual_scale = max(float(moe_cfg.pop("residual_scale", 0.35)), 0.0)
        # Keep the residual modules compatible with one checkpoint contract,
        # but do not optimize router balancing during wheel-only pretraining.
        self.moe_auxiliary_enabled = bool(
            moe_cfg.pop("auxiliary_enabled", self.residual_scale > 0.0)
        )
        self.action_smoothing = float(moe_cfg.pop("action_smoothing", 0.45))
        self.action_smoothing = min(max(self.action_smoothing, 0.0), 1.0)
        self.action_rate_limit = max(float(moe_cfg.pop("action_rate_limit", 0.35)), 0.0)
        self.router_entropy_coef = max(float(moe_cfg.pop("router_entropy_coef", 0.002)), 0.0)
        self.router_balance_coef = max(float(moe_cfg.pop("router_balance_coef", 0.01)), 0.0)
        self.router_min_usage_coef = max(float(moe_cfg.pop("router_min_usage_coef", 0.02)), 0.0)
        self.router_min_usage = min(max(float(moe_cfg.pop("router_min_usage", 0.03)), 0.0), 0.25)
        self.router_usage_ema_decay = min(max(float(moe_cfg.pop("router_usage_ema_decay", 0.98)), 0.0), 0.9999)
        router_hidden_dims = tuple(int(value) for value in moe_cfg.pop("router_hidden_dims", (128, 64)))
        expert_hidden_dims = tuple(int(value) for value in moe_cfg.pop("expert_hidden_dims", hidden_dims))
        wheel_bias = float(moe_cfg.pop("wheel_router_bias", 2.5))
        if moe_cfg:
            raise ValueError(f"Unknown MoE model options: {sorted(moe_cfg)}")

        # ``mlp`` remains the single-policy path for old checkpoints and for
        # the 12-DoF MiniDog task.  Go2W enables the residual MoE in its RL
        # config without changing the surrounding runner or action manager.
        if self.use_moe:
            self.wheel_policy = MLP(fused_dim, self.distribution.input_dim, hidden_dims, activation)
            self.router = MLP(fused_dim, len(self.expert_names), router_hidden_dims, activation)
            self.residual_experts = nn.ModuleList(
                MLP(fused_dim, self.distribution.input_dim, expert_hidden_dims, activation)
                for _ in self.expert_names[1:]
            )
            self.distribution.init_mlp_weights(self.wheel_policy)
            self.distribution.init_mlp_weights(self.router)
            for expert in self.residual_experts:
                self.distribution.init_mlp_weights(expert)
                # Start with a pure wheel policy.  PPO can grow residuals as
                # the terrain curriculum exposes useful obstacle signals.
                last_linear = next(
                    layer for layer in reversed(expert) if isinstance(layer, nn.Linear)
                )
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
            router_output = next(layer for layer in reversed(self.router) if isinstance(layer, nn.Linear))
            nn.init.zeros_(router_output.weight)
            nn.init.constant_(router_output.bias, -wheel_bias / (len(self.expert_names) - 1))
            router_output.bias.data[0] = wheel_bias
        else:
            self.mlp = MLP(fused_dim, self.distribution.input_dim, hidden_dims, activation)
            self.distribution.init_mlp_weights(self.mlp)

    def _current_proprio(self, history: torch.Tensor) -> torch.Tensor:
        pieces = []
        offset = 0
        for term_dim in self.proprio_term_dims:
            term = history[:, offset : offset + term_dim * self.proprio_history_length]
            term = term.reshape(history.shape[0], self.proprio_history_length, term_dim)
            pieces.append(term[:, -1])
            offset += term_dim * self.proprio_history_length
        return torch.cat(pieces, dim=-1)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        proprio = obs[self.proprio_group]
        radar = obs[self.radar_group]
        self._last_radar_valid_fraction = (radar < 0.999).float().mean().detach()
        current_proprio = self._current_proprio(proprio)
        state_latent = self.state_encoder(proprio)
        terrain_latent = self.terrain_encoder(radar, state_latent)
        return torch.cat((state_latent, terrain_latent, current_proprio), dim=-1)

    def _moe_action(self, latent: torch.Tensor, current_proprio: torch.Tensor) -> torch.Tensor:
        """Blend obstacle residuals while keeping wheel locomotion as the base."""
        wheel_action = self.wheel_policy(latent)
        router_logits = self.router(latent) / self.router_temperature
        router_weights = torch.softmax(router_logits, dim=-1)
        self._last_router_weights = router_weights
        with torch.no_grad():
            usage = router_weights.detach().mean(dim=0)
            self.expert_usage_ema.mul_(self.router_usage_ema_decay).add_(
                usage, alpha=1.0 - self.router_usage_ema_decay
            )
        # Keep the wheel action in every mixture and select only the highest
        # scoring residual expert.  The two selected router weights are
        # renormalized, so the result remains a continuous weighted blend;
        # action smoothing/rate limiting handles expert-boundary transitions.
        residual_logits = router_logits[:, 1:]
        residual_index = torch.argmax(residual_logits, dim=-1)
        # ``top_k=2`` is wheel plus one residual expert.  Evaluate each
        # residual only for the samples routed to it instead of evaluating all
        # three residual MLPs for every sample.
        if self.moe_top_k != 2:
            raise ValueError("Go2W residual MoE currently supports top_k=2 only")
        selected_residual = torch.zeros_like(wheel_action)
        for expert_id, expert in enumerate(self.residual_experts):
            sample_ids = torch.nonzero(residual_index == expert_id, as_tuple=False).squeeze(-1)
            if sample_ids.numel() > 0:
                selected_residual = selected_residual.index_copy(
                    0, sample_ids, expert(latent.index_select(0, sample_ids))
                )
        selected_weight = router_weights[:, 1:].gather(1, residual_index[:, None]).squeeze(1)
        wheel_weight = router_weights[:, 0]
        blend_norm = (wheel_weight + selected_weight).clamp_min(1.0e-6)
        residual_weight = selected_weight / blend_norm
        target_action = wheel_action + residual_weight[:, None] * self.residual_scale * torch.tanh(
            selected_residual
        )

        # The previous action is already part of the proprioceptive frame. A
        # stateless limit is export-friendly and still prevents expert weight
        # changes from producing an abrupt motor command.
        last_action_dim = min(self.action_dim, current_proprio.shape[-1])
        # This policy is an integrator: a_t = a_{t-1} + delta, so the previous
        # *sampled* action is its integration state.  Without a bound that state
        # random-walks for the whole episode; the unified run reached
        # mean_action_acc 132 and action_rate -40/step.  The bound applies to
        # training and export alike, so sim2sim sees the same contract.
        last_action = current_proprio[:, -last_action_dim:].clamp(
            -ACTION_ABSOLUTE_LIMIT, ACTION_ABSOLUTE_LIMIT
        )
        if last_action.shape[-1] != target_action.shape[-1]:
            last_action = torch.zeros_like(target_action)
        delta = self.action_smoothing * (target_action - last_action)
        if self.action_rate_limit > 0.0:
            delta = self.action_rate_limit * torch.tanh(delta / self.action_rate_limit)
        self._last_diagnostics = {
            "wheel_expert_weight": router_weights[:, 0].detach().mean(),
            "step_expert_weight": router_weights[:, 1].detach().mean(),
            "gap_expert_weight": router_weights[:, 2].detach().mean(),
            "recovery_expert_weight": router_weights[:, 3].detach().mean(),
            "residual_norm": torch.linalg.vector_norm((target_action - wheel_action).detach(), dim=-1).mean(),
            "top_residual_weight": residual_weight.detach().mean(),
            "action_delta": torch.linalg.vector_norm(delta.detach(), dim=-1).mean(),
            "wheel_usage_ema": self.expert_usage_ema[0].detach(),
            "step_usage_ema": self.expert_usage_ema[1].detach(),
            "gap_usage_ema": self.expert_usage_ema[2].detach(),
            "recovery_usage_ema": self.expert_usage_ema[3].detach(),
            "radar_valid_fraction": self._last_radar_valid_fraction,
        }
        return (last_action + delta).clamp(-ACTION_ABSOLUTE_LIMIT, ACTION_ABSOLUTE_LIMIT)

    def get_moe_auxiliary_loss(self) -> torch.Tensor:
        """Return differentiable router regularization for the current batch.

        The wheel expert is intentionally allowed to dominate. Balancing is
        applied only to residual experts, preventing one obstacle expert from
        monopolizing the router without forcing obstacle behavior on flat
        terrain.
        """
        if not self.use_moe or not self.moe_auxiliary_enabled or self._last_router_weights is None:
            return torch.zeros((), device=next(self.parameters()).device)
        weights = self._last_router_weights
        residual_usage = weights[:, 1:].mean(dim=0)
        residual_target = residual_usage.mean().detach()
        balance = torch.mean(torch.square(residual_usage - residual_target))
        min_usage = torch.mean(torch.relu(self.router_min_usage - residual_usage).square())
        entropy = -torch.sum(weights * torch.log(weights.clamp_min(1.0e-6)), dim=-1).mean()
        loss = (
            -self.router_entropy_coef * entropy
            + self.router_balance_coef * balance
            + self.router_min_usage_coef * min_usage
        )
        self._last_diagnostics.update(
            {
                "router_entropy": entropy.detach(),
                "residual_usage_balance": balance.detach(),
                "router_min_usage_penalty": min_usage.detach(),
                "moe_auxiliary_loss": loss.detach(),
            }
        )
        return loss

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        current_proprio = latent[:, -self.proprio_frame_dim :]
        if self.use_moe:
            mlp_output = self._moe_action(latent, current_proprio)
        else:
            mlp_output = self.mlp(latent)
        if stochastic_output:
            self.distribution.update(mlp_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_output)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: Any = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> None:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def get_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return detached MoE statistics for the custom runner logger."""
        return dict(self._last_diagnostics)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """Migrate the first radar convolution from pre-mask checkpoints."""
        key = "terrain_encoder.convolution.0.weight"
        source = state_dict.get(key)
        target = self.state_dict().get(key)
        if source is not None and target is not None and source.shape != target.shape:
            if source.ndim == 4 and target.ndim == 4 and source.shape[0:1] == target.shape[0:1]:
                migrated = target.detach().clone()
                channels = min(source.shape[1], target.shape[1])
                migrated[:, :channels] = source[:, :channels]
                state_dict = dict(state_dict)
                state_dict[key] = migrated
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def update_normalization(self, obs: TensorDict) -> None:
        del obs

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

