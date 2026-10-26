"""Lightweight checkpoint catalog for the UI model picker.

The catalog does not load Torch checkpoints just to draw the list; it scans
the dedicated ``sim2sim/model`` directory and records path, size, mtime and an
iteration parsed from ``model_<N>.pt``.  The actual load happens only when the
user confirms a selection, so opening the page stays fast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT

_ITER_RE = re.compile(r"model_(\d+)", re.IGNORECASE)


def _workspace_root() -> Path:
    """Find D:\\mujoco_test from either sim2sim or sim2sim_next.

    In the source copy ``PROJECT_ROOT.parent`` is the workspace; in the
    installed original it is ``mujoco`` and the workspace is two levels up.
    """

    parent = PROJECT_ROOT.parent
    for candidate in (parent, parent.parent, parent.parent.parent):
        if (candidate / "HighTorque-MiniDog").exists():
            return candidate
        try:
            if any(candidate.glob("model_*.pt")):
                return candidate
        except OSError:
            pass
    return parent


@dataclass(frozen=True)
class ModelEntry:
    path: Path
    name: str
    size_bytes: int
    mtime: float
    iteration: int | None
    group: str

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024.0 * 1024.0)

    @property
    def date_text(self) -> str:
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")

    @property
    def iteration_text(self) -> str:
        return "-" if self.iteration is None else str(self.iteration)

    def label(self) -> str:
        return f"{self.name}   iter={self.iteration_text}   {self.size_mb:.1f} MB   {self.date_text}"


def _dedupe(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


class ModelCatalog:
    def __init__(self, roots: list[Path] | None = None, limit: int = 120) -> None:
        self.limit = int(limit)
        if roots is None:
            roots = self._default_roots()
        self.roots = _dedupe([Path(root) for root in roots])
        self.entries: list[ModelEntry] = []
        self.errors: list[str] = []

    @staticmethod
    def _default_roots() -> list[Path]:
        # Only the dedicated model folder is exposed in the UI.  This prevents
        # accidental selection of old MJLab checkpoints with incompatible
        # contracts and makes the picker's contents predictable.
        return [PROJECT_ROOT / "model"]

    def _scan_root(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        if root.is_file():
            return [root] if root.suffix.lower() == ".pt" else []
        # The root is shallow for the workspace, recursive for model trees.
        recursive = root.name in {"model", "run", "models", "training_logs", "training_logs_success"}
        pattern = "**/*.pt" if recursive else "*.pt"
        try:
            return list(root.glob(pattern))
        except OSError as exc:
            self.errors.append(f"{root}: {exc}")
            return []

    def scan(self) -> list[ModelEntry]:
        self.errors.clear()
        entries: list[ModelEntry] = []
        for root in self.roots:
            for path in self._scan_root(root):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_size <= 0:
                    continue
                match = _ITER_RE.search(path.stem)
                iteration = int(match.group(1)) if match else None
                group = root.name if root.is_dir() else root.parent.name
                entries.append(
                    ModelEntry(
                        path=path.resolve(),
                        name=path.stem,
                        size_bytes=int(stat.st_size),
                        mtime=float(stat.st_mtime),
                        iteration=iteration,
                        group=group,
                    )
                )
        # The dedicated model/ folder first, then workspace-root checkpoints,
        # then training trees; within a group prefer higher iteration and newer.
        model_dir = (PROJECT_ROOT / "model").resolve()
        root_dir = _workspace_root().resolve()

        def priority(item: ModelEntry) -> int:
            parent = item.path.parent
            if parent == model_dir or model_dir in item.path.parents:
                return 0
            if parent == root_dir:
                return 1
            return 2

        entries.sort(
            key=lambda item: (
                priority(item),
                -(item.iteration if item.iteration is not None else -1),
                -item.mtime,
            )
        )
        self.entries = entries[: self.limit]
        return self.entries

    def find(self, path: str | Path) -> ModelEntry | None:
        target = Path(path).resolve()
        for entry in self.entries:
            if entry.path == target:
                return entry
        return None
