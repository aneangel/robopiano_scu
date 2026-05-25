#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for _path in (REPO_ROOT, REPO_ROOT / "Intermezzo" / "src", REPO_ROOT / "partita" / "src"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def onset_frames(target_keys: np.ndarray, threshold: float) -> list[int]:
    active = np.asarray(target_keys, dtype=np.float32)[:, :88] > float(threshold)
    previous = np.zeros((88,), dtype=bool)
    frames: list[int] = []
    for frame, row in enumerate(active):
        if bool(np.any(row & ~previous)):
            frames.append(int(frame))
        previous = row
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavotte: keep planned hand states only around target onset windows.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pre-substeps", type=int, default=0)
    parser.add_argument("--post-substeps", type=int, default=6)
    parser.add_argument("--neutral-source", choices=("first", "zero"), default="first")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.source_npz, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    source_dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32)
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    substeps = max(int(source_dense.shape[0] // max(target_keys.shape[0], 1)), 1)
    dense_dt = 0.05 / float(substeps)
    if str(args.neutral_source) == "zero":
        neutral = np.zeros((source_dense.shape[1],), dtype=np.float32)
    else:
        neutral = source_dense[0].astype(np.float32)
    dense = np.repeat(neutral.reshape(1, -1), source_dense.shape[0], axis=0).astype(np.float32)
    keep = np.zeros((source_dense.shape[0],), dtype=bool)
    for frame in onset_frames(target_keys, float(args.threshold)):
        start = max(int(frame) * substeps - max(int(args.pre_substeps), 0), 0)
        end = min(int(frame) * substeps + max(int(args.post_substeps), 1), source_dense.shape[0])
        if start < end:
            dense[start:end] = source_dense[start:end]
            keep[start:end] = True
    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["planned_hand_joints"] = dense[::substeps][: target_keys.shape[0]].astype(np.float32)
    payload["planned_hand_velocities"] = compute_hand_velocities(payload["planned_hand_joints"], control_timestep=0.05)
    payload["gavotte_kept_dense_frames"] = keep.astype(np.float32)
    atomic_save_npz(out / "trajectory.npz", **payload)

    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(args.environment_name),
        demo_id=0,
        actions=np.zeros((dense.shape[0], 39), dtype=np.float32),
        goals=dense_goals,
        hand_joints=dense,
        environment_name=str(args.environment_name),
    )
    sim_summary = simulate_rp1m_rollout(
        traj,
        RolloutConfig(
            mode="hand_state",
            dataset_timestep=float(dense_dt),
            simulation_timestep=float(dense_dt),
            hand_anchor_y_offset=None,
            hand_state_action_source="zero",
            restore_initial_hand=True,
            set_hand_qvel=False,
            threshold=float(args.threshold),
            render_mp4=False,
            render_audio=False,
        ),
        out / "rp1m_sim",
    )
    with np.load(sim_summary["rollout_npz"], allow_pickle=False) as rollout:
        played = np.asarray(rollout["source_played_piano"], dtype=np.float32)
        goals = np.asarray(rollout["goals"], dtype=np.float32)
    score = score_rollout(
        target_keys=goals,
        played_keys=played,
        dt=float(dense_dt),
        threshold=float(args.threshold),
        timing_tolerance_s=0.15,
    )
    result = {
        "source_npz": str(args.source_npz),
        "output_dir": str(out),
        "pre_substeps": int(args.pre_substeps),
        "post_substeps": int(args.post_substeps),
        "kept_dense_frames": int(np.count_nonzero(keep)),
        "onset_windows": int(len(onset_frames(target_keys, float(args.threshold)))),
        "event_f1": float(event_f1(score)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "played": int(score.get("played_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
