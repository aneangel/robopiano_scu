from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


TIE_BREAK_METRICS = (
    "piano/missed_events",
    "piano/false_events",
    "piano/timing_abs_error_mean_s",
    "fingertip/active_l2_mean",
    "tracking/joint_pos_rmse",
    "control/action_clip_rate",
)

LEADERBOARD_COLUMNS = (
    "rank",
    "checkpoint",
    "config",
    "primary_metric",
    "primary_mean",
    "piano/event_f1_mean",
    "piano/missed_events_mean",
    "piano/false_events_mean",
    "piano/timing_abs_error_mean_s_mean",
    "fingertip/active_l2_mean_mean",
    "tracking/joint_pos_rmse_mean",
    "control/action_clip_rate_mean",
    "num_episodes",
    "eval_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Etude checkpoint rollout evaluations.")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--primary-metric", default="piano/event_f1")
    parser.add_argument("--selection-mode", choices=("max", "min"), default="max")
    parser.add_argument("--output", default=None)
    parser.add_argument("--copy-best-to", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    output = Path(args.output) if args.output else eval_root / "leaderboard.csv"
    rows = rank_checkpoints(
        eval_root,
        primary_metric=args.primary_metric,
        selection_mode=args.selection_mode,
    )
    if not rows:
        raise ValueError(f"No aggregate_metrics.json files found under {eval_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEADERBOARD_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    best_payload = {
        "checkpoint": best["checkpoint"],
        "eval_dir": best["eval_dir"],
        "primary_metric": args.primary_metric,
        "selection_mode": args.selection_mode,
        "metrics": {key: best[key] for key in LEADERBOARD_COLUMNS if key.endswith("_mean")},
    }
    (output.parent / "best_checkpoint.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.copy_best_to:
        source = Path(str(best["checkpoint"]))
        if not source.exists():
            raise FileNotFoundError(f"Winning checkpoint does not exist and cannot be copied: {source}")
        destination = Path(args.copy_best_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"Wrote leaderboard to {output}")
    print(f"Best checkpoint: {best['checkpoint']}")


def rank_checkpoints(
    eval_root: Path,
    *,
    primary_metric: str = "piano/event_f1",
    selection_mode: str = "max",
) -> list[dict[str, Any]]:
    rows = [_row_from_aggregate(path, primary_metric) for path in sorted(eval_root.rglob("aggregate_metrics.json"))]
    rows.sort(key=lambda row: _sort_key(row, primary_metric=primary_metric, selection_mode=selection_mode))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _row_from_aggregate(path: Path, primary_metric: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    row = {
        "rank": 0,
        "checkpoint": payload.get("checkpoint", ""),
        "config": payload.get("config", ""),
        "primary_metric": primary_metric,
        "primary_mean": _metric_mean(metrics, primary_metric),
        "num_episodes": payload.get("num_episodes", _metric_count(metrics, primary_metric)),
        "eval_dir": str(path.parent),
    }
    for metric in ("piano/event_f1", *TIE_BREAK_METRICS):
        row[f"{metric}_mean"] = _metric_mean(metrics, metric)
    return row


def _sort_key(row: dict[str, Any], *, primary_metric: str, selection_mode: str) -> tuple[Any, ...]:
    primary = _coerce_float(row.get("primary_mean"))
    primary_key = float("inf") if primary is None else (-primary if selection_mode == "max" else primary)
    tie_keys = []
    for metric in TIE_BREAK_METRICS:
        value = _coerce_float(row.get(f"{metric}_mean"))
        tie_keys.append(float("inf") if value is None else value)
    return (primary_key, *tie_keys, str(row.get("checkpoint", "")), str(row.get("eval_dir", "")))


def _metric_mean(metrics: dict[str, Any], metric: str) -> float | str:
    value = metrics.get(metric, {})
    if isinstance(value, dict) and "mean" in value:
        return value["mean"]
    return ""


def _metric_count(metrics: dict[str, Any], metric: str) -> int | str:
    value = metrics.get(metric, {})
    if isinstance(value, dict) and "count" in value:
        return value["count"]
    return ""


def _coerce_float(value: Any) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
