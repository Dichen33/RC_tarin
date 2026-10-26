"""Policy backend contract used by the simulation loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PolicyContract:
    architecture: str
    action_dim: int
    proprio_dim: int
    history_length: int
    frame_dim: int
    scan_dim: int
    scan_size: tuple[float, float]
    checkpoint: Path | None = None
    note: str = ""


class PolicyBackend:
    """Inference-only policy.

    ``proprio_history`` is term-major flattened, exactly as the training
    ``actor_proprio`` group: for every term, all history frames, then the next
    term.  ``height_scan`` is the current radar frame.
    """

    contract: PolicyContract

    def reset(self) -> None:
        raise NotImplementedError

    def act(
        self,
        proprio_history: np.ndarray,
        height_scan: np.ndarray,
        scan_history: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        c = self.contract
        return (
            f"{c.architecture} action={c.action_dim} proprio={c.proprio_dim} "
            f"history={c.history_length}x{c.frame_dim} scan={c.scan_dim} "
            f"scan_size={c.scan_size}"
        )
