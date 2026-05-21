#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sweep_common import DEFAULT_SWEEP_ROOT, SWEEP_KEYS, adaptive_objective, is_balanced_target_hit, json_dump


SORT_KEY = lambda row: (
    -(row.get("frame_f1") or -1.0),
    -(row.get("frame_precision") or -1.0),
    -(row.get("matched_press_events") or -1.0),
    row.get("FP") if row.get("FP") is not None else 10**9,
)

ARCHITECTURE_CHANGES = [
    "Sequence-level/Viterbi finger assignment instead of greedy per-waypoint assignment.",
    "Explicit inactive-finger clearance constraint in IK.",
    "Wrong-key proximity avoidance term in IK.",
    "Press-window-specific active attraction instead of global magnetism.",
    "Separate hover trajectory and press trajectory.",
    "Add press-lead/timing calibration learned from lag_sweep.",
    "Replace pure qpos interpolation with minimum-jerk fingertip trajectory.",
    "Train a PD + residual controller where the planner prioritizes safe high-precision hover geometry and the controller handles final actuation.",
    "Add a learned correction model for fingertip target offsets from played-vs-target errors.",
    "Use global trajectory optimization over the whole phrase instead of independent anchor IK.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate dense sweep results.")
    parser.add_argument("--output-root", default=str(DEFAULT_SWEEP_ROOT))
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_run(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config.json"
    score_path = run_dir / "score.json"
    render_path = run_dir / "render_summary.json"
    metrics_path = run_dir / "metrics.json"
    lag_path = run_dir / "lag_sweep.json"
    if not (config_path.is_file() and score_path.is_file() and render_path.is_file()):
        return None
    config = _read_json(config_path)
    score = _read_json(score_path)
    render = _read_json(render_path)
    metrics = _read_json(metrics_path) if metrics_path.is_file() else {}
    lag = _read_json(lag_path) if lag_path.is_file() else {}
    best_lag = lag.get("best") or {}
    row = {
        "run_name": run_dir.name,
        "frame_f1": score.get("frame_f1"),
        "frame_precision": score.get("frame_precision"),
        "frame_recall": score.get("frame_recall"),
        "TP": score.get("frame_true_positives"),
        "FP": score.get("frame_false_positives"),
        "FN": score.get("frame_false_negatives"),
        "matched_press_events": score.get("matched_press_events"),
        "missed_key_presses": score.get("missed_key_presses"),
        "mispresses": score.get("mispresses"),
        "timing_abs_error_p95_s": score.get("timing_abs_error_p95_s"),
        "best_lag_s": best_lag.get("lag_s"),
        "best_lag_f1": best_lag.get("frame_f1"),
        "trajectory_ik_anchor_success_rate": metrics.get("ik_anchor_success_rate"),
        "render_dir": render.get("run_dir"),
        "adaptive_objective": None,
        "meets_balanced_target": False,
    }
    for key in SWEEP_KEYS:
        row[key] = config.get(key)
    row["adaptive_objective"] = adaptive_objective(row)
    row["meets_balanced_target"] = is_balanced_target_hit(row)
    return row


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    run_dirs = [path for path in output_root.iterdir() if path.is_dir() and (path / "config.json").is_file()]
    rows = [row for row in (_row_from_run(path) for path in run_dirs) if row is not None]
    rows.sort(key=SORT_KEY)
    csv_path = output_root / "sweep_results.csv"
    fieldnames = [
        "run_name",
        "frame_f1",
        "frame_precision",
        "frame_recall",
        "TP",
        "FP",
        "FN",
        "matched_press_events",
        "missed_key_presses",
        "mispresses",
        "timing_abs_error_p95_s",
        "best_lag_s",
        "best_lag_f1",
        "adaptive_objective",
        "meets_balanced_target",
        "trajectory_ik_anchor_success_rate",
        "render_dir",
        *SWEEP_KEYS,
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    top10 = rows[:10]
    summary_path = output_root / "top10_summary.json"
    json_dump(summary_path, {"top10": top10})
    recommendation = {"targets_met": False, "recommendations": []}
    best_balanced = next(
        (
            row for row in rows
            if (row.get("frame_f1") or 0.0) >= 0.60 and (row.get("frame_precision") or 0.0) >= 0.60
        ),
        None,
    )
    if best_balanced is None:
        recommendation["recommendations"] = ARCHITECTURE_CHANGES
    else:
        recommendation["targets_met"] = True
        recommendation["best_run"] = best_balanced["run_name"]
    json_dump(output_root / "architectural_recommendation.json", recommendation)

    print("Top 10 runs:")
    for index, row in enumerate(top10, start=1):
        print(
            f"{index:02d}. {row['run_name']} "
            f"F1={row['frame_f1']:.3f} "
            f"P={row['frame_precision']:.3f} "
            f"R={row['frame_recall']:.3f} "
            f"TP={row['TP']} FP={row['FP']} matched={row['matched_press_events']} "
            f"obj={row['adaptive_objective']:.3f}"
        )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
