#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _event_count(binary: np.ndarray) -> int:
    values = np.asarray(binary, dtype=bool).reshape(-1)
    if values.size == 0:
        return 0
    starts = values & np.concatenate([[True], ~values[:-1]])
    return int(np.count_nonzero(starts))


def _summarize_frames(mask: np.ndarray, target: np.ndarray, played: np.ndarray) -> dict[str, object]:
    if not np.any(mask):
        return {"frames": 0}
    t = target[mask]
    p = played[mask]
    fp = np.logical_and(~t, p)
    fn = np.logical_and(t, ~p)
    tp = np.logical_and(t, p)
    return {
        "frames": int(np.count_nonzero(mask)),
        "tp_frames": int(np.count_nonzero(tp)),
        "fp_frames": int(np.count_nonzero(fp)),
        "fn_frames": int(np.count_nonzero(fn)),
        "fp_keys_top": [
            {"key": int(key), "frames": int(count)}
            for key, count in sorted(
                [(key, count) for key, count in enumerate(fp.sum(axis=0).astype(int)) if count],
                key=lambda item: (-item[1], item[0]),
            )[:15]
        ],
        "played_press_events": int(sum(_event_count(p[:, key]) for key in range(p.shape[1]))),
        "target_press_events": int(sum(_event_count(t[:, key]) for key in range(t.shape[1]))),
        "fp_press_events": int(sum(_event_count(fp[:, key]) for key in range(fp.shape[1]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--rollout-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--waypoint-window", type=int, default=2)
    args = parser.parse_args()

    traj = np.load(Path(args.trajectory_npz).expanduser(), allow_pickle=False)
    rollout = np.load(Path(args.rollout_npz).expanduser(), allow_pickle=False)
    target = np.asarray(rollout["target_keys"], dtype=np.float32)[:, :88] > float(args.threshold)
    played = np.asarray(rollout["played_keys"], dtype=np.float32)[:, :88] > float(args.threshold)
    steps = min(target.shape[0], played.shape[0])
    target = target[:steps]
    played = played[:steps]

    control_steps = int(np.asarray(traj["target_keys"]).shape[0])
    substeps = max(int(np.asarray(traj["planned_hand_joints_dense"]).shape[0] // max(control_steps, 1)), 1)
    waypoint_frames = np.asarray(traj["waypoint_frames"], dtype=np.int64).reshape(-1)
    waypoint_dense = waypoint_frames * substeps
    near_waypoint = np.zeros((steps,), dtype=bool)
    for frame in waypoint_dense:
        start = max(int(frame) - int(args.waypoint_window), 0)
        end = min(int(frame) + int(args.waypoint_window) + 1, steps)
        near_waypoint[start:end] = True
    active_target = np.any(target, axis=1)
    inactive_target = ~active_target
    fp_any = np.any(np.logical_and(~target, played), axis=1)
    categories = {
        "all": np.ones((steps,), dtype=bool),
        "near_waypoint": near_waypoint,
        "away_from_waypoint": ~near_waypoint,
        "target_active": active_target,
        "target_inactive": inactive_target,
        "fp_any_near_waypoint": near_waypoint & fp_any,
        "fp_any_away_from_waypoint": (~near_waypoint) & fp_any,
    }
    report = {
        "trajectory_npz": str(Path(args.trajectory_npz).expanduser()),
        "rollout_npz": str(Path(args.rollout_npz).expanduser()),
        "steps": int(steps),
        "substeps": int(substeps),
        "waypoint_window": int(args.waypoint_window),
        "summaries": {name: _summarize_frames(mask, target, played) for name, mask in categories.items()},
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
