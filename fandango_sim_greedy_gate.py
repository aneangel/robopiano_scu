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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_rows(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    rows = []
    for row in summary.get("injection_rows", []):
        start = int(row.get("dense_start", -1))
        end = int(row.get("dense_end", -1))
        if start >= 0 and end > start:
            rows.append({**row, "dense_start": start, "dense_end": end})
    rows.sort(key=lambda row: (int(row.get("target_frame", 0)), int(row["dense_start"]), int(row["dense_end"])))
    return rows


def simulate_score(
    *,
    payload: dict[str, np.ndarray],
    dense: np.ndarray,
    output_dir: Path,
    environment_name: str,
    threshold: float,
    dense_dt: float,
) -> dict[str, Any]:
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    substeps = max(int(dense.shape[0] // max(target_keys.shape[0], 1)), 1)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(environment_name),
        demo_id=0,
        actions=np.zeros((dense.shape[0], 39), dtype=np.float32),
        goals=dense_goals,
        hand_joints=np.asarray(dense, dtype=np.float32),
        environment_name=str(environment_name),
    )
    try:
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
                threshold=float(threshold),
                render_mp4=False,
                render_audio=False,
            ),
            output_dir,
        )
        with np.load(sim_summary["rollout_npz"], allow_pickle=False) as rollout:
            played = np.asarray(rollout["source_played_piano"], dtype=np.float32)
            goals = np.asarray(rollout["goals"], dtype=np.float32)
        score = score_rollout(
            target_keys=goals,
            played_keys=played,
            dt=float(dense_dt),
            threshold=float(threshold),
            timing_tolerance_s=0.15,
        )
        return {
            "event_f1": float(event_f1(score)),
            "frame_f1": float(score.get("frame_f1", 0.0)),
            "matched": int(score.get("matched_press_events", 0)),
            "target": int(score.get("target_press_events", 0)),
            "played": int(score.get("played_press_events", 0)),
            "mispresses": int(score.get("mispresses", 0)),
        }
    except Exception as exc:
        return {
            "event_f1": -1.0,
            "frame_f1": -1.0,
            "matched": 0,
            "target": int(target_keys.shape[0]),
            "played": 10**9,
            "mispresses": 10**9,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fandango: simulator-greedy pulse selection for Impromptu injections.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-delta-event-f1", type=float, default=1e-6)
    parser.add_argument("--max-evals", type=int, default=0)
    parser.add_argument("--keep-if-matched-increases", action="store_true")
    parser.add_argument("--matched-gain-max-f1-drop", type=float, default=0.0)
    parser.add_argument("--base-mode", choices=("source", "neutral"), default="source")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = load_npz(Path(args.source_npz))
    candidate = load_npz(Path(args.candidate_npz))
    rows = load_rows(Path(args.candidate_summary))
    source_dense = np.asarray(source["planned_hand_joints_dense"], dtype=np.float32)
    candidate_dense = np.asarray(candidate["planned_hand_joints_dense"], dtype=np.float32)
    if source_dense.shape != candidate_dense.shape:
        raise ValueError(f"dense shape mismatch: source {source_dense.shape} candidate {candidate_dense.shape}")
    dense = source_dense.copy()
    control_steps = int(np.asarray(source["target_keys"]).shape[0])
    substeps = max(int(dense.shape[0] // max(control_steps, 1)), 1)
    dense_dt = 0.05 / float(substeps)
    if str(args.base_mode) == "neutral":
        neutral = source_dense[0].copy()
        dense = np.repeat(neutral.reshape(1, -1), dense.shape[0], axis=0).astype(np.float32)

    current = simulate_score(
        payload=source,
        dense=dense,
        output_dir=out / "eval_000_baseline",
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        dense_dt=float(dense_dt),
    )
    initial = dict(current)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    max_evals = int(args.max_evals)
    if max_evals <= 0:
        max_evals = len(rows)
    for eval_index, row in enumerate(rows[:max_evals], start=1):
        start = int(row["dense_start"])
        end = int(row["dense_end"])
        previous_slice = dense[start:end].copy()
        dense[start:end] = candidate_dense[start:end]
        trial = simulate_score(
            payload=source,
            dense=dense,
            output_dir=out / f"eval_{eval_index:03d}",
            environment_name=str(args.environment_name),
            threshold=float(args.threshold),
            dense_dt=float(dense_dt),
        )
        delta = float(trial["event_f1"] - current["event_f1"])
        matched_gain = int(trial["matched"] - current["matched"])
        keep = delta >= float(args.min_delta_event_f1)
        if (
            bool(args.keep_if_matched_increases)
            and matched_gain > 0
            and trial["event_f1"] >= current["event_f1"] - max(float(args.matched_gain_max_f1_drop), 0.0) - 1e-9
        ):
            keep = True
        record = {
            **row,
            "eval_index": int(eval_index),
            "trial": trial,
            "previous": current,
            "delta_event_f1": float(delta),
            "delta_matched": int(matched_gain),
        }
        if keep:
            current = trial
            accepted.append(record)
        else:
            dense[start:end] = previous_slice
            rejected.append(record)

    payload = dict(source)
    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["planned_hand_joints"] = dense[::substeps][:control_steps].astype(np.float32)
    payload["planned_hand_velocities"] = compute_hand_velocities(
        payload["planned_hand_joints"], control_timestep=0.05
    ).astype(np.float32)
    mask = np.zeros((dense.shape[0],), dtype=np.float32)
    for row in accepted:
        mask[int(row["dense_start"]) : int(row["dense_end"])] = 1.0
    payload["fandango_accepted_dense_frames"] = mask
    atomic_save_npz(out / "trajectory.npz", **payload)
    final_score = simulate_score(
        payload=payload,
        dense=dense,
        output_dir=out / "rp1m_sim",
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        dense_dt=float(dense_dt),
    )
    result = {
        "source_npz": str(args.source_npz),
        "candidate_npz": str(args.candidate_npz),
        "candidate_summary": str(args.candidate_summary),
        "output_dir": str(out),
        "rows": int(len(rows)),
        "evaluated": int(min(max_evals, len(rows))),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "initial": initial,
        "greedy_score": current,
        "final_score": final_score,
        "matched_gain_max_f1_drop": float(args.matched_gain_max_f1_drop),
        "base_mode": str(args.base_mode),
        "accepted": accepted[:400],
        "rejected": rejected[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
