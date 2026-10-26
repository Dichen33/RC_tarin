"""Ordered Go2W training stages and their transfer policy.

The table is deliberately data-only so the automatic trainer and reporting
tools use the same task order and do not silently disagree about what a stage
means.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageProfile:
    name: str
    task: str
    max_iterations: int
    learning_rate: float
    transfer_mode: str
    critic_transfer: bool
    history_revisit: float
    evaluation_tasks: tuple[str, ...]


STAGE_PROFILES: tuple[StageProfile, ...] = (
    StageProfile("flat", "Robot-Go2W-Flat-v0", 15000, 2.0e-4, "full_resume", True, 0.0, ("Robot-Go2W-Flat-v0",)),
    StageProfile("rough", "Robot-Go2W-Rough-v0", 15000, 2.0e-4, "actor_critic_reset_optimizer", True, 0.10, ("Robot-Go2W-Flat-v0", "Robot-Go2W-Rough-v0")),
    StageProfile("step", "Robot-Go2W-Step-v0", 12000, 1.5e-4, "actor_critic_reset_optimizer", True, 0.10, ("Robot-Go2W-Flat-v0", "Robot-Go2W-Rough-v0", "Robot-Go2W-Step-v0")),
    StageProfile("gap", "Robot-Go2W-Gap-v0", 18000, 1.5e-4, "actor_only_reset_critic_optimizer", False, 0.10, ("Robot-Go2W-Flat-v0", "Robot-Go2W-Rough-v0", "Robot-Go2W-Step-v0", "Robot-Go2W-Gap-v0")),
    StageProfile("high_wall", "Robot-Go2W-HighWall-v0", 20000, 1.2e-4, "actor_only_reset_critic_optimizer", False, 0.10, ("Robot-Go2W-Step-v0", "Robot-Go2W-Gap-v0", "Robot-Go2W-HighWall-v0")),
    StageProfile("plum_blossom", "Robot-Go2W-PlumBlossom-v0", 22000, 1.2e-4, "actor_only_reset_critic_optimizer", False, 0.10, ("Robot-Go2W-Rough-v0", "Robot-Go2W-Step-v0", "Robot-Go2W-Gap-v0", "Robot-Go2W-PlumBlossom-v0")),
    StageProfile("mixed", "Robot-Go2W-Mixed-v0", 30000, 1.0e-4, "actor_critic_reset_optimizer", True, 0.25, ("Robot-Go2W-Flat-v0", "Robot-Go2W-Rough-v0", "Robot-Go2W-Step-v0", "Robot-Go2W-Gap-v0", "Robot-Go2W-HighWall-v0", "Robot-Go2W-PlumBlossom-v0", "Robot-Go2W-Mixed-v0")),
)

STAGE_BY_NAME = {profile.name: profile for profile in STAGE_PROFILES}


def stage_names() -> tuple[str, ...]:
    return tuple(profile.name for profile in STAGE_PROFILES)

