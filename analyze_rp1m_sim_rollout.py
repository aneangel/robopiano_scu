#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_run(run_dir: Path, threshold: float) -> dict[str, object]:
    rollout_path = run_dir / "rollout.npz"
    score_path = run_dir / "intermezzo_score.json"
    with np.load(rollout_path, allow_pickle=False) as data:
        goals = np.asarray(data["goals"], dtype=np.float32)[:, :88]
        played = np.asarray(data["source_played_piano"], dtype=np.float32)[:, :88]
    steps = min(goals.shape[0], played.shape[0])
    goals = goals[:steps] > float(threshold)
    played = played[:steps] > float(threshold)
    target_active = goals.any(axis=1)
    played_active = played.any(axis=1)
    fp = np.logical_and(~goals, played)
    fn = np.logical_and(goals, ~played)
    tp = np.logical_and(goals, played)
    fp_frame = fp.any(axis=1)
    no_target_fp_frames = np.logical_and(~target_active, fp_frame)
    active_target_fp_frames = np.logical_and(target_active, fp_frame)
    target_only_frames = target_active.sum()
    no_target_frames = (~target_active).sum()
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    return {
        "run": run_dir.name,
        "steps": int(steps),
        "target_active_frames": int(target_only_frames),
        "no_target_frames": int(no_target_frames),
        "played_active_frames": int(played_active.sum()),
        "tp_cells": int(tp.sum()),
        "fp_cells": int(fp.sum()),
        "fn_cells": int(fn.sum()),
        "fp_frames": int(fp_frame.sum()),
        "no_target_fp_frames": int(no_target_fp_frames.sum()),
        "active_target_fp_frames": int(active_target_fp_frames.sum()),
        "no_target_fp_cell_count": int(fp[~target_active].sum()),
        "active_target_fp_cell_count": int(fp[target_active].sum()),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    root = Path(args.root)
    rows = [analyze_run(p, args.threshold) for p in sorted(root.iterdir()) if (p / "rollout.npz").exists()]
    totals = {
        "runs": rows,
        "total_fp_cells": int(sum(int(row["fp_cells"]) for row in rows)),
        "total_no_target_fp_cells": int(sum(int(row["no_target_fp_cell_count"]) for row in rows)),
        "total_active_target_fp_cells": int(sum(int(row["active_target_fp_cell_count"]) for row in rows)),
        "total_fp_frames": int(sum(int(row["fp_frames"]) for row in rows)),
        "total_no_target_fp_frames": int(sum(int(row["no_target_fp_frames"]) for row in rows)),
        "total_active_target_fp_frames": int(sum(int(row["active_target_fp_frames"]) for row in rows)),
    }
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
