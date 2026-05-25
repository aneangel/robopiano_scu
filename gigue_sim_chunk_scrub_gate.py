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


def simulate_score(payload: dict[str, np.ndarray], dense: np.ndarray, out: Path, env: str, threshold: float, dense_dt: float) -> dict[str, Any]:
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    substeps = max(int(dense.shape[0] // max(target_keys.shape[0], 1)), 1)
    goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(env),
        demo_id=0,
        actions=np.zeros((dense.shape[0], 39), dtype=np.float32),
        goals=goals,
        hand_joints=dense,
        environment_name=str(env),
    )
    sim = simulate_rp1m_rollout(
        traj,
        RolloutConfig(
            mode="hand_state",
            dataset_timestep=float(dense_dt),
            simulation_timestep=float(dense_dt),
            hand_anchor_y_offset=None,
            hand_state_action_source="zero",
            restore_initial_hand=True,
            set_hand_qvel=False,
            threshold=float(threshold),
            render_mp4=False,
            render_audio=False,
        ),
        out,
    )
    with np.load(sim["rollout_npz"], allow_pickle=False) as data:
        played = np.asarray(data["source_played_piano"], dtype=np.float32)
        target = np.asarray(data["goals"], dtype=np.float32)
    score = score_rollout(target_keys=target, played_keys=played, dt=float(dense_dt), threshold=float(threshold), timing_tolerance_s=0.15)
    return {
        "event_f1": float(event_f1(score)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "played": int(score.get("played_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gigue: simulator-greedy neutral chunk scrub for false-contact reduction.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-control-frames", type=int, default=5)
    parser.add_argument("--min-delta-event-f1", type=float, default=1e-6)
    parser.add_argument("--max-evals", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.source_npz, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32).copy()
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control_steps = int(target_keys.shape[0])
    substeps = max(int(dense.shape[0] // max(control_steps, 1)), 1)
    dense_dt = 0.05 / float(substeps)
    neutral = dense[0].astype(np.float32)
    current = simulate_score(payload, dense, out / "eval_000_baseline", str(args.environment_name), float(args.threshold), dense_dt)
    initial = dict(current)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    chunk = max(int(args.chunk_control_frames), 1)
    starts = list(range(0, control_steps, chunk))
    if int(args.max_evals) > 0:
        starts = starts[: int(args.max_evals)]
    scrubbed = np.zeros((dense.shape[0],), dtype=np.float32)
    for eval_index, control_start in enumerate(starts, start=1):
        control_end = min(control_start + chunk, control_steps)
        start = int(control_start) * substeps
        end = min(int(control_end) * substeps, dense.shape[0])
        previous = dense[start:end].copy()
        dense[start:end] = neutral
        trial = simulate_score(payload, dense, out / f"eval_{eval_index:03d}", str(args.environment_name), float(args.threshold), dense_dt)
        delta = float(trial["event_f1"] - current["event_f1"])
        record = {
            "eval_index": int(eval_index),
            "control_start": int(control_start),
            "control_end": int(control_end),
            "dense_start": int(start),
            "dense_end": int(end),
            "previous": current,
            "trial": trial,
            "delta_event_f1": float(delta),
        }
        if delta >= float(args.min_delta_event_f1):
            current = trial
            scrubbed[start:end] = 1.0
            accepted.append(record)
        else:
            dense[start:end] = previous
            rejected.append(record)

    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["planned_hand_joints"] = dense[::substeps][:control_steps].astype(np.float32)
    payload["planned_hand_velocities"] = compute_hand_velocities(payload["planned_hand_joints"], control_timestep=0.05)
    payload["gigue_scrubbed_dense_frames"] = scrubbed
    atomic_save_npz(out / "trajectory.npz", **payload)
    final_score = simulate_score(payload, dense, out / "rp1m_sim", str(args.environment_name), float(args.threshold), dense_dt)
    result = {
        "source_npz": str(args.source_npz),
        "output_dir": str(out),
        "chunk_control_frames": int(chunk),
        "initial": initial,
        "greedy_score": current,
        "final_score": final_score,
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "accepted": accepted[:400],
        "rejected": rejected[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
