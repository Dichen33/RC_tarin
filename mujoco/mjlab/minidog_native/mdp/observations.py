from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def gait_phase(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.03,
    stationary_frequency: float = 0.0,
    base_frequency: float = 1.8,
    max_frequency: float = 2.8,
    speed_scale: float = 1.0,
) -> torch.Tensor:
    """Return a gait clock as ``[sin(phi), cos(phi)]``.

    The phase is hidden from the action stream and is only exposed as an
    observation. When the command is near zero, the clock slows down so the
    policy can treat standing separately from locomotion.
    """

    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = command_norm > command_threshold
    moving_frequency = torch.clamp(
        base_frequency + speed_scale * command_norm,
        min=base_frequency,
        max=max_frequency,
    )
    frequency = torch.where(
        active,
        moving_frequency,
        torch.full_like(command_norm, stationary_frequency),
    )

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=command_norm.dtype)
    phase = (env.episode_length_buf.float() * env.step_dt * frequency + env_ids * 0.17320508) % 1.0
    angle = 2.0 * math.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)

