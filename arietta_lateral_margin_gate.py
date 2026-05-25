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


RIGHT_FOREARM_TX = 21
RIGHT_FOREARM_TY = 22
LEFT_FOREARM_TX = 44
LEFT_FOREARM_TY = 45


def event_f1_from_score(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def compact_score(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_f1": float(event_f1_from_score(score)),
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


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def simulate_score(
    payload: dict[str, np.ndarray],
    dense: np.ndarray,
    out: Path,
    env: str,
    threshold: float,
    dense_dt: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return compact_score(score), score


def forearm_indices_for_key(key: int, split_key: int) -> tuple[int, int]:
    if int(key) < int(split_key):
        return LEFT_FOREARM_TX, LEFT_FOREARM_TY
    return RIGHT_FOREARM_TX, RIGHT_FOREARM_TY


def main() -> None:
    parser = argparse.ArgumentParser(description="Arietta: lateral forearm margin sweep for adjacent-key false contacts.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-key", type=int, default=48)
    parser.add_argument("--pre-dense-frames", type=int, default=2)
    parser.add_argument("--post-dense-frames", type=int, default=10)
    parser.add_argument("--ty-deltas", default="-0.024,-0.016,-0.010,-0.006,0.006,0.010,0.016,0.024")
    parser.add_argument("--tx-deltas", default="0.0")
    parser.add_argument("--active-target-only", action="store_true")
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--max-total-evals", type=int, default=180)
    parser.add_argument("--min-delta-event-f1", type=float, default=1e-6)
    parser.add_argument("--allow-equal-f1-mispress-drop", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source_npz)
    payload = load_npz(source_path)
    dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32).copy()
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control_steps = int(target_keys.shape[0])
    substeps = max(int(dense.shape[0] // max(control_steps, 1)), 1)
    dense_dt = 0.05 / float(substeps)
    active_dense = np.repeat(target_keys > float(args.threshold), substeps, axis=0)[: dense.shape[0], :88]

    ty_deltas = parse_float_list(str(args.ty_deltas))
    tx_deltas = parse_float_list(str(args.tx_deltas))
    deltas = [(tx, ty) for tx in tx_deltas for ty in ty_deltas if abs(tx) > 0.0 or abs(ty) > 0.0]
    if not deltas:
        raise ValueError("at least one non-zero tx/ty delta is required")

    current, detail = simulate_score(payload, dense, out / "eval_000_baseline", str(args.environment_name), float(args.threshold), dense_dt)
    initial = dict(current)
    events: list[dict[str, int]] = []
    for item in detail.get("mispress_events", []):
        frame = int(item.get("frame", -1))
        key = int(item.get("key", -1))
        if key < 0 or key >= 88 or frame < 0 or frame >= dense.shape[0]:
            continue
        if bool(args.active_target_only):
            if not bool(np.any(active_dense[frame])):
                continue
            if bool(active_dense[frame, key]):
                continue
        events.append({"frame": frame, "key": key})
        if int(args.max_events) > 0 and len(events) >= int(args.max_events):
            break

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    eval_count = 0
    max_total = int(args.max_total_evals)
    selected_delta = np.zeros((dense.shape[0], 2), dtype=np.float32)

    for event_index, event in enumerate(events):
        if max_total > 0 and eval_count >= max_total:
            break
        frame = int(event["frame"])
        key = int(event["key"])
        start = max(frame - max(int(args.pre_dense_frames), 0), 0)
        end = min(frame + max(int(args.post_dense_frames), 0) + 1, dense.shape[0])
        if end <= start:
            continue
        tx_index, ty_index = forearm_indices_for_key(key, int(args.split_key))
        previous = dense[start:end].copy()
        best_score: dict[str, Any] | None = None
        best_slice: np.ndarray | None = None
        best_delta: tuple[float, float] | None = None
        trials: list[dict[str, Any]] = []
        for tx_delta, ty_delta in deltas:
            if max_total > 0 and eval_count >= max_total:
                break
            dense[start:end] = previous
            dense[start:end, tx_index] = dense[start:end, tx_index] + np.float32(tx_delta)
            dense[start:end, ty_index] = dense[start:end, ty_index] + np.float32(ty_delta)
            eval_count += 1
            trial, _ = simulate_score(
                payload,
                dense,
                out / f"eval_{eval_count:04d}",
                str(args.environment_name),
                float(args.threshold),
                dense_dt,
            )
            trials.append({"tx_delta": float(tx_delta), "ty_delta": float(ty_delta), "trial": trial})
            if best_score is None or score_key(trial) > score_key(best_score):
                best_score = dict(trial)
                best_slice = dense[start:end].copy()
                best_delta = (float(tx_delta), float(ty_delta))
        dense[start:end] = previous
        delta_f1 = float(best_score["event_f1"] - current["event_f1"]) if best_score else 0.0
        accept = bool(best_score) and delta_f1 >= float(args.min_delta_event_f1)
        if (
            bool(best_score)
            and bool(args.allow_equal_f1_mispress_drop)
            and delta_f1 >= -1e-12
            and int(best_score["mispresses"]) < int(current["mispresses"])
        ):
            accept = True
        record = {
            "event_index": int(event_index),
            "event": dict(event),
            "dense_start": int(start),
            "dense_end": int(end),
            "previous": dict(current),
            "best_delta": None if best_delta is None else {"tx": best_delta[0], "ty": best_delta[1]},
            "best_trial": best_score,
            "delta_event_f1": float(delta_f1),
            "trials": trials,
        }
        if accept and best_slice is not None and best_score is not None:
            dense[start:end] = best_slice
            if best_delta is not None:
                selected_delta[start:end, 0] = np.float32(best_delta[0])
                selected_delta[start:end, 1] = np.float32(best_delta[1])
            current = best_score
            accepted.append(record)
        else:
            rejected.append(record)

    output_payload = dict(payload)
    output_payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    output_payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    output_payload["planned_hand_joints"] = dense[::substeps][:control_steps].astype(np.float32)
    output_payload["planned_hand_velocities"] = compute_hand_velocities(output_payload["planned_hand_joints"], control_timestep=0.05)
    output_payload["arietta_selected_delta"] = selected_delta.astype(np.float32)
    atomic_save_npz(out / "trajectory.npz", **output_payload)
    final_score, _ = simulate_score(payload, dense, out / "rp1m_sim", str(args.environment_name), float(args.threshold), dense_dt)
    result = {
        "source_npz": str(source_path),
        "output_dir": str(out),
        "ty_deltas": ty_deltas,
        "tx_deltas": tx_deltas,
        "active_target_only": bool(args.active_target_only),
        "initial": initial,
        "greedy_score": current,
        "final_score": final_score,
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "event_count": int(len(events)),
        "eval_count": int(eval_count),
        "accepted": accepted[:400],
        "rejected": rejected[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
