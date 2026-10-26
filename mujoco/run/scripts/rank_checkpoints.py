from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _float(row, key)
        if value is not None:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank checkpoints using metrics.csv.")
    parser.add_argument("run_dir", type=Path, help="Training run directory containing metrics.csv")
    parser.add_argument("--csv", type=Path, default=None, help="Optional metrics.csv path")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    csv_path = args.csv or (args.run_dir / "metrics.csv")
    if not csv_path.exists():
        raise SystemExit(f"metrics.csv not found: {csv_path}")

    rows = list(csv.DictReader(csv_path.open("r", newline="", encoding="utf-8")))
    candidates = []
    for row in rows:
        iteration = _float(row, "iteration")
        reward = _float(row, "mean_reward")
        progress = _float(row, "Episode_Reward/course_completion")
        gap = _first_float(
            row,
            (
                "Curriculum/terrain_levels/flat_gap_success_rate",
                "Curriculum/terrain_levels/flat_gap_success_rate_batch",
            ),
        )
        plum = _first_float(
            row,
            (
                "Curriculum/terrain_levels/rough_plum_blossom_success_rate",
                "Curriculum/terrain_levels/rough_plum_blossom_success_rate_batch",
            ),
        )
        wall = _first_float(
            row,
            (
                "Curriculum/terrain_levels/flat_high_wall_success_rate",
                "Curriculum/terrain_levels/flat_high_wall_success_rate_batch",
            ),
        )
        if iteration is None:
            continue
        # TensorBoard also stores time-axis scalars at non-iteration steps.
        # A checkpoint row must have the PPO training summary reward.
        if reward is None:
            continue
        checkpoint = args.run_dir / f"model_{int(iteration)}.pt"
        if not checkpoint.exists():
            continue

        # Prefer obstacle success; use the later iteration as a deterministic tie-breaker.
        run_name = args.run_dir.name.lower()
        if "high_wall" in run_name or "wall" in run_name:
            score = wall if wall is not None else 0.0
        elif "gap" in run_name:
            score = gap if gap is not None else 0.0
        else:
            score = (
                0.40 * (gap if gap is not None else 0.0)
                + 0.40 * (plum if plum is not None else 0.0)
                + 0.20 * (wall if wall is not None else 0.0)
            )
        candidates.append((score, iteration, reward, progress, gap, plum, wall))

    candidates.sort(reverse=True)
    print("score,iteration,mean_reward,completion,gap_success,plum_success,wall_success,checkpoint")
    for score, iteration, reward, progress, gap, plum, wall in candidates[: args.top]:
        checkpoint = args.run_dir / f"model_{int(iteration)}.pt"
        print(
            f"{score:.4f},{int(iteration)},{reward if reward is not None else ''},"
            f"{progress if progress is not None else ''},{gap if gap is not None else ''},"
            f"{plum if plum is not None else ''},{wall if wall is not None else ''},{checkpoint}"
        )


if __name__ == "__main__":
    main()
