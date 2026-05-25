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


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    p = matched / played if played else 0.0
    r = matched / target if target else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Retest Impromptu with zero-order-hold dense hand states.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    control = np.asarray(payload["planned_hand_joints"], dtype=np.float32)
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    base_dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32)
    substeps = max(int(base_dense.shape[0] // max(control.shape[0], 1)), 1)
    dense = np.repeat(control, substeps, axis=0).astype(np.float32)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    payload["planned_hand_joints_dense"] = dense
    atomic_save_npz(out / "trajectory.npz", **payload)
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
    score = score_rollout(target_keys=goals, played_keys=played, dt=float(dense_dt), threshold=float(args.threshold), timing_tolerance_s=0.15)
    result = {
        "source_npz": str(source),
        "output_dir": str(out),
        "substeps": int(substeps),
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
