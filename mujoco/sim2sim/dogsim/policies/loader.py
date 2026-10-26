"""Checkpoint inspection and policy construction.

The loader is deliberately strict: a checkpoint whose observation/action
contract does not match the selected robot fails at startup, instead of
silently feeding a misordered observation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from ..config import PolicyConfig, RunConfig
from .ame import AMEActor, AMEPolicy
from .base import PolicyBackend, PolicyContract
from .mlp import MLPActor, MLPPolicy


def _actor_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    actor = state.get("actor_state_dict")
    if actor is None:
        raise ValueError("Checkpoint has no 'actor_state_dict'")
    actor = dict(actor)
    if actor and all(key.startswith("actor.") for key in actor):
        actor = {key[len("actor.") :]: value for key, value in actor.items()}
    return actor


def _infer_map_shape(scan_dim: int, scan_size: tuple[float, float] | None, resolution: float) -> tuple[int, int]:
    if scan_size is not None:
        nx = int(scan_size[0] / resolution + 1.0e-6) + 1
        ny = int(scan_size[1] / resolution + 1.0e-6) + 1
        if nx * ny == scan_dim:
            return nx, ny
    known = {403: (31, 13), 216: (18, 12)}
    if scan_dim in known:
        return known[scan_dim]
    root = int(round(math.sqrt(scan_dim)))
    for height in range(root, 0, -1):
        if scan_dim % height == 0:
            width = scan_dim // height
            # Prefer a forward-long map, like every training checkpoint.
            return (max(height, width), min(height, width))
    raise ValueError(f"Cannot infer a height-map shape for scan_dim={scan_dim}")


def _resolve_scan_size(cfg: RunConfig, scan_dim: int) -> tuple[float, float]:
    if cfg.policy.scan_size is not None:
        return tuple(cfg.policy.scan_size)
    return cfg.robot.scan_size_by_dim.get(scan_dim, cfg.robot.scan_size)


def _check_dim(name: str, actual: int, expected: int) -> None:
    if int(actual) != int(expected):
        raise ValueError(f"{name} mismatch: checkpoint={actual}, robot/config={expected}")


def load_policy(cfg: RunConfig) -> PolicyBackend:
    checkpoint = cfg.policy.checkpoint
    if checkpoint is None:
        raise ValueError("policy.checkpoint is required")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if cfg.policy.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("policy.device=cuda requested but torch.cuda.is_available() is false")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a dict: {checkpoint}")
    actor_state = _actor_state(state)

    is_moe = "wheel_policy.0.weight" in actor_state and "router.0.weight" in actor_state
    is_ame = "state_encoder.0.weight" in actor_state
    architecture = "moe" if is_moe else "ame" if is_ame else "mlp"
    requested = cfg.policy.architecture
    if requested != "auto" and requested != architecture:
        raise ValueError(
            f"policy.architecture={requested!r} but checkpoint is {architecture!r}"
        )

    term_dims = tuple(int(v) for v in cfg.robot.observation_term_dims)
    frame_dim = int(sum(term_dims))
    if frame_dim <= 0:
        raise ValueError("robot.observation_term_dims must sum to a positive frame dimension")

    history = int(cfg.policy.history_length or cfg.robot.history_length)
    if history < 1:
        raise ValueError("history length must be >= 1")

    if architecture in {"moe", "ame"}:
        proprio_dim = int(actor_state["state_encoder.0.weight"].shape[1])
        if proprio_dim % frame_dim == 0:
            inferred_history = proprio_dim // frame_dim
        else:
            inferred_history = history
        if inferred_history != history:
            # The checkpoint is authoritative: the observation history must
            # match the temporal encoder that was trained.
            history = inferred_history
        _check_dim("proprioception history", proprio_dim, history * frame_dim)
        action_dim = int(actor_state["wheel_policy.6.weight"].shape[0] if is_moe else actor_state["mlp.6.weight"].shape[0])
        _check_dim("action dimension", action_dim, cfg.robot.action_dim)

        conv = actor_state["terrain_encoder.convolution.0.weight"]
        in_channels = int(conv.shape[1])
        if in_channels not in {3, 4}:
            raise ValueError(f"Unsupported radar encoder input channels: {in_channels}")
        use_return_mask = in_channels == 4
        if cfg.policy.use_return_mask is not None and bool(cfg.policy.use_return_mask) != use_return_mask:
            raise ValueError(
                f"policy.use_return_mask={cfg.policy.use_return_mask} but checkpoint conv has "
                f"{in_channels} input channels"
            )

        scan_dim = int(cfg.policy.scan_dim or cfg.robot.scan_dim)
        scan_size = _resolve_scan_size(cfg, scan_dim)
        if cfg.policy.scan_size is not None and tuple(cfg.policy.scan_size) != tuple(scan_size):
            raise ValueError(
                f"policy.scan_size={cfg.policy.scan_size} conflicts with the checkpoint "
                f"scan geometry {scan_size} for scan_dim={scan_dim}"
            )
        map_shape = tuple(int(v) for v in cfg.policy.map_shape)
        if map_shape[0] * map_shape[1] != scan_dim:
            map_shape = _infer_map_shape(scan_dim, scan_size, cfg.scan_resolution)
        if map_shape[0] * map_shape[1] != scan_dim:
            raise ValueError(
                f"map_shape={map_shape} product does not match scan_dim={scan_dim}"
            )

        actor = AMEActor(
            action_dim=action_dim,
            proprio_frame_dim=frame_dim,
            history_length=history,
            term_dims=term_dims,
            map_shape=map_shape,
            use_return_mask=use_return_mask,
            moe=is_moe,
            residual_scale=cfg.policy.residual_scale,
            action_smoothing=cfg.policy.action_smoothing,
            action_rate_limit=cfg.policy.action_rate_limit,
            action_absolute_limit=cfg.policy.action_absolute_limit,
            router_temperature=cfg.policy.router_temperature,
        )
        try:
            actor.load_state_dict(actor_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint actor does not match the inferred {architecture.upper()} contract: {exc}"
            ) from exc
        contract = PolicyContract(
            architecture=architecture,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            history_length=history,
            frame_dim=frame_dim,
            scan_dim=scan_dim,
            scan_size=scan_size,
            checkpoint=checkpoint,
            note=f"map_shape={map_shape} return_mask={use_return_mask}",
        )
        backend: PolicyBackend = AMEPolicy(actor, contract, device=cfg.policy.device)
    else:
        obs_dim = int(actor_state["mlp.0.weight"].shape[1])
        if obs_dim % history:
            raise ValueError(f"MLP observation dimension {obs_dim} is not divisible by history {history}")
        single_obs_dim = obs_dim // history
        scan_dim = single_obs_dim - frame_dim
        if scan_dim <= 0:
            raise ValueError(
                f"MLP single-frame observation {single_obs_dim} is smaller than the {frame_dim} proprio terms"
            )
        action_dim = int(actor_state["mlp.6.weight"].shape[0])
        _check_dim("action dimension", action_dim, cfg.robot.action_dim)
        scan_size = _resolve_scan_size(cfg, scan_dim)
        if cfg.policy.scan_size is not None and tuple(cfg.policy.scan_size) != tuple(scan_size):
            raise ValueError(
                f"policy.scan_size={cfg.policy.scan_size} conflicts with the checkpoint "
                f"scan geometry {scan_size} for scan_dim={scan_dim}"
            )
        actor = MLPActor(obs_dim, action_dim)
        try:
            actor.load_state_dict(actor_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Checkpoint actor does not match the MLP contract: {exc}") from exc
        contract = PolicyContract(
            architecture="mlp",
            action_dim=action_dim,
            proprio_dim=history * frame_dim,
            history_length=history,
            frame_dim=frame_dim,
            scan_dim=scan_dim,
            scan_size=scan_size,
            checkpoint=checkpoint,
            note="legacy flattened history",
        )
        backend = MLPPolicy(
            actor,
            contract,
            base_dim=frame_dim,
            scan_dim=scan_dim,
            action_clip=5.0,
            device=cfg.policy.device,
        )

    iter_value = state.get("iter")
    print(
        f"[Policy] {checkpoint}\n"
        f"[Policy] architecture={contract.architecture} action={contract.action_dim} "
        f"proprio={contract.proprio_dim} history={contract.history_length}x{contract.frame_dim} "
        f"scan={contract.scan_dim} scan_size={contract.scan_size} "
        f"{'iter=' + str(iter_value) if iter_value is not None else ''}"
    )
    print(f"[Policy] {contract.note}")
    return backend
