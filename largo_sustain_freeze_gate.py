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


def compact(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_f1": float(event_f1(score)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "played": int(score.get("played_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
    }


def score_key(score: dict[str, Any]) -> tuple[float, int, int, float]:
    return (
        float(score["event_f1"]),
        -int(score["mispresses"]),
        int(score["matched"]),
        float(score["frame_f1"]),
    )


def simulate_score(
    payload: dict[str, np.ndarray],
    dense: np.ndarray,
    out: Path,
    env: str,
    threshold: float,
    dense_dt: float,
) -> dict[str, Any]:
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    substeps = max(int(dense.shape[0] // max(target_keys.shape[0], 1)), 1)
    goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(env),
        demo_id=0,
        actions=np.zeros((dense.shape[0], 39), dtype=np.float32),
        goals=goals,
        hand_joints=np.asarray(dense, dtype=np.float32),
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
        played = np.asarray(data["source_played_piano"], dtype=np.float32)[:, :88]
        target = np.asarray(data["goals"], dtype=np.float32)[:, :88]
    score = score_rollout(
        target_keys=target,
        played_keys=played,
        dt=float(dense_dt),
        threshold=float(threshold),
        timing_tolerance_s=0.15,
    )
    return compact(score)


def identical_active_runs(target_keys: np.ndarray, threshold: float, min_control_frames: int) -> list[tuple[int, int, tuple[int, ...]]]:
    active = np.asarray(target_keys, dtype=np.float32)[:, :88] > float(threshold)
    runs: list[tuple[int, int, tuple[int, ...]]] = []
    start = 0
    for frame in range(1, active.shape[0] + 1):
        if frame == active.shape[0] or not np.array_equal(active[frame], active[start]):
            keys = tuple(int(v) for v in np.flatnonzero(active[start]).tolist())
            if keys and frame - start >= max(int(min_control_frames), 2):
                runs.append((int(start), int(frame), keys))
            start = int(frame)
    return runs


def freeze_interval(
    dense: np.ndarray,
    *,
    dense_start: int,
    dense_end: int,
    mode: str,
) -> np.ndarray:
    previous = np.asarray(dense[dense_start:dense_end], dtype=np.float32)
    if previous.size == 0:
        return previous.copy()
    if mode == "first":
        anchor = previous[0]
        return np.repeat(anchor.reshape(1, -1), previous.shape[0], axis=0).astype(np.float32)
    if mode == "middle":
        anchor = previous[previous.shape[0] // 2]
        return np.repeat(anchor.reshape(1, -1), previous.shape[0], axis=0).astype(np.float32)
    if mode == "control_zoh":
        # Preserve the first control row in the sustained region but keep the
        # within-control dense substep shape from that row.
        return np.repeat(previous[:1], previous.shape[0], axis=0).astype(np.float32)
    raise ValueError(f"unknown freeze mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Largo: simulator-greedy freeze of sustained identical target masks.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-control-frames", type=int, default=2)
    parser.add_argument("--mode", choices=("first", "middle", "control_zoh"), default="first")
    parser.add_argument("--max-evals", type=int, default=0)
    parser.add_argument("--min-delta-event-f1", type=float, default=1e-6)
    parser.add_argument("--allow-equal-f1-mispress-drop", action="store_true")
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

    current = simulate_score(payload, dense, out / "eval_000_baseline", str(args.environment_name), float(args.threshold), dense_dt)
    initial = dict(current)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    freeze_mask = np.zeros((dense.shape[0],), dtype=np.float32)
    runs = identical_active_runs(target_keys, float(args.threshold), int(args.min_control_frames))
    if int(args.max_evals) > 0:
        runs = runs[: int(args.max_evals)]

    for eval_index, (control_start, control_end, keys) in enumerate(runs, start=1):
        dense_start = int(control_start) * substeps
        dense_end = min(int(control_end) * substeps, dense.shape[0])
        previous = dense[dense_start:dense_end].copy()
        dense[dense_start:dense_end] = freeze_interval(
            dense,
            dense_start=dense_start,
            dense_end=dense_end,
            mode=str(args.mode),
        )
        trial = simulate_score(
            payload,
            dense,
            out / f"eval_{eval_index:03d}",
            str(args.environment_name),
            float(args.threshold),
            dense_dt,
        )
        delta = float(trial["event_f1"] - current["event_f1"])
        accept = delta >= float(args.min_delta_event_f1)
        if (
            bool(args.allow_equal_f1_mispress_drop)
            and delta >= -1e-12
            and int(trial["mispresses"]) < int(current["mispresses"])
        ):
            accept = True
        record = {
            "eval_index": int(eval_index),
            "control_start": int(control_start),
            "control_end": int(control_end),
            "dense_start": int(dense_start),
            "dense_end": int(dense_end),
            "keys": list(keys),
            "previous": current,
            "trial": trial,
            "delta_event_f1": float(delta),
        }
        if accept:
            current = trial
            freeze_mask[dense_start:dense_end] = 1.0
            accepted.append(record)
        else:
            dense[dense_start:dense_end] = previous
            rejected.append(record)

    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["planned_hand_joints"] = dense[::substeps][:control_steps].astype(np.float32)
    payload["planned_hand_velocities"] = compute_hand_velocities(payload["planned_hand_joints"], control_timestep=0.05)
    payload["largo_frozen_dense_frames"] = freeze_mask
    atomic_save_npz(out / "trajectory.npz", **payload)
    final_score = simulate_score(payload, dense, out / "rp1m_sim", str(args.environment_name), float(args.threshold), dense_dt)
    result = {
        "source_npz": str(args.source_npz),
        "output_dir": str(out),
        "mode": str(args.mode),
        "min_control_frames": int(args.min_control_frames),
        "initial": initial,
        "greedy_score": current,
        "final_score": final_score,
        "runs_considered": int(len(runs)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "accepted": accepted[:400],
        "rejected": rejected[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
