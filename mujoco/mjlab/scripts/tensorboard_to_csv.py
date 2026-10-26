from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _column(tag: str) -> str:
    mapping = {
        "Train/mean_reward": "mean_reward",
        "Train/mean_episode_length": "mean_episode_length",
        "Policy/mean_std": "mean_action_std",
        "Loss/learning_rate": "learning_rate",
        "Perf/total_fps": "fps",
        "Perf/collection_time": "collection_time",
        "Perf/learning_time": "learning_time",
        "Loss/value": "loss_value",
        "Loss/surrogate": "loss_surrogate",
        "Loss/entropy": "loss_entropy",
    }
    return mapping.get(tag, tag)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MJLab TensorBoard scalars to metrics.csv.")
    parser.add_argument("run_dir", type=Path, help="Training run directory")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    event_files = sorted(args.run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"No TensorBoard event file found in {args.run_dir}")

    values: dict[int, dict[str, float | int]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                row = values.setdefault(int(event.step), {"iteration": int(event.step)})
                row[_column(tag)] = float(event.value)

    output = args.output or (args.run_dir / "metrics.csv")
    rows = [values[step] for step in sorted(values)]
    fieldnames = ["iteration"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
