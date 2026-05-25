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
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def interp_rows(anchor_x: np.ndarray, anchor_y: np.ndarray, out_x: np.ndarray) -> np.ndarray:
    anchor_x = np.asarray(anchor_x, dtype=np.float64).reshape(-1)
    anchor_y = np.asarray(anchor_y, dtype=np.float32)
    out_x = np.asarray(out_x, dtype=np.float64).reshape(-1)
    out = np.empty((out_x.size, anchor_y.shape[1]), dtype=np.float32)
    for col in range(anchor_y.shape[1]):
        out[:, col] = np.interp(out_x, anchor_x, anchor_y[:, col]).astype(np.float32)
    return out


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-npz", required=True)
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    baseline_path = Path(args.baseline_npz)
    candidate_path = Path(args.candidate_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(baseline_path, allow_pickle=False) as base_data:
        payload = {key: np.asarray(base_data[key]) for key in base_data.files}
        base_waypoints = np.asarray(base_data["waypoint_hand_joints"], dtype=np.float32)
    with np.load(candidate_path, allow_pickle=False) as cand_data:
        cand_waypoints = np.asarray(cand_data["waypoint_hand_joints"], dtype=np.float32)
    alpha = float(args.alpha)
    blended = ((1.0 - alpha) * base_waypoints + alpha * cand_waypoints).astype(np.float32)
    waypoint_frames = np.asarray(payload["waypoint_frames"], dtype=np.int64)
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control_steps = int(np.asarray(payload["planned_hand_joints"]).shape[0])
    substeps = max(int(np.asarray(payload["planned_hand_joints_dense"]).shape[0] // max(control_steps, 1)), 1)
    control_x = np.arange(control_steps, dtype=np.float64)
    dense_x = np.arange(control_steps * substeps, dtype=np.float64) / float(substeps)
    control_qpos = interp_rows(waypoint_frames, blended, control_x)
    dense_qpos = interp_rows(control_x, control_qpos, dense_x)
    payload["waypoint_hand_joints"] = blended
    payload["planned_hand_joints"] = control_qpos.astype(np.float32)
    payload["planned_hand_joints_dense"] = dense_qpos.astype(np.float32)
    atomic_save_npz(out / "trajectory.npz", **payload)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(args.environment_name),
        demo_id=0,
        actions=np.zeros((dense_qpos.shape[0], 39), dtype=np.float32),
        goals=dense_goals,
        hand_joints=dense_qpos,
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
    score = score_rollout(target_keys=goals, played_keys=played, dt=float(dense_dt), threshold=float(args.threshold), timing_tolerance_s=0.15)
    result = {
        "baseline_npz": str(baseline_path),
        "candidate_npz": str(candidate_path),
        "output_dir": str(out),
        "alpha": alpha,
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
