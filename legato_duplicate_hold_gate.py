#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for _path in (
    REPO_ROOT,
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402
from impromptu.joint_space_trajectory import BAGATELLE_FINGER_JOINT_INDEX_ROWS  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


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


def nearest_fingers_for_key(
    kin: BagatelleKinematics,
    qpos: np.ndarray,
    key: int,
    top_fingers: int,
) -> list[dict[str, Any]]:
    fingertips = np.asarray(kin.fingertip_positions_for_qpos(qpos), dtype=np.float32)
    target = np.asarray(kin.key_contact_targets(np.asarray([int(key)], dtype=np.int32))[0], dtype=np.float32)
    distances = np.linalg.norm(fingertips[:, :2] - target[:2].reshape(1, 2), axis=1)
    order = np.argsort(distances, kind="stable")[: max(int(top_fingers), 1)]
    return [
        {
            "finger": int(finger),
            "xy_distance": float(distances[int(finger)]),
        }
        for finger in order.tolist()
    ]


def active_interval(active_dense: np.ndarray, frame: int, key: int) -> tuple[int, int]:
    values = np.asarray(active_dense[:, int(key)], dtype=bool)
    start = int(frame)
    while start > 0 and bool(values[start - 1]):
        start -= 1
    end = int(frame) + 1
    while end < values.size and bool(values[end]):
        end += 1
    return int(start), int(end)


def reference_frame_for_mode(
    *,
    mode: str,
    interval_start: int,
    interval_end: int,
    event_frame: int,
    substeps: int,
    dense_total: int,
) -> int:
    if mode == "onset":
        return int(interval_start)
    if mode == "onset_plus":
        return int(min(interval_start + max(int(substeps) - 1, 0), dense_total - 1))
    if mode == "mid_interval":
        return int(min(interval_start + max((interval_end - interval_start) // 2, 0), dense_total - 1))
    if mode == "pre_50ms":
        return int(max(interval_start, event_frame - max(int(substeps), 1)))
    if mode == "pre_100ms":
        return int(max(interval_start, event_frame - 2 * max(int(substeps), 1)))
    raise ValueError(f"unknown mode {mode!r}")


def apply_finger_hold(
    dense: np.ndarray,
    *,
    start: int,
    end: int,
    finger: int,
    reference_qpos: np.ndarray,
) -> np.ndarray:
    candidate = dense[start:end].copy()
    joints = np.asarray(BAGATELLE_FINGER_JOINT_INDEX_ROWS[int(finger)], dtype=np.int64)
    candidate[:, joints] = np.asarray(reference_qpos, dtype=np.float32)[joints].reshape(1, -1)
    return candidate.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Legato: hold target-active fingers to suppress duplicate restrikes.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pre-dense-frames", type=int, default=10)
    parser.add_argument("--post-dense-frames", type=int, default=8)
    parser.add_argument("--top-fingers", type=int, default=2)
    parser.add_argument("--modes", default="onset,onset_plus,pre_50ms,pre_100ms,mid_interval")
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--max-total-evals", type=int, default=180)
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
    active_dense = np.repeat(target_keys > float(args.threshold), substeps, axis=0)[: dense.shape[0], :88]
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]

    current, detail = simulate_score(
        payload,
        dense,
        out / "eval_000_baseline",
        str(args.environment_name),
        float(args.threshold),
        dense_dt,
    )
    initial = dict(current)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    eval_count = 0
    held = np.zeros((dense.shape[0],), dtype=np.float32)
    duplicate_events = [
        event
        for event in detail.get("mispress_events", [])
        if 0 <= int(event.get("key", -1)) < 88
        and 0 <= int(event.get("frame", -1)) < active_dense.shape[0]
        and bool(active_dense[int(event.get("frame", -1)), int(event.get("key", -1))])
    ]

    config = BagatelleConfig(environment_name=str(args.environment_name), threshold=float(args.threshold), seed=0)
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        for event_index, event in enumerate(duplicate_events[: max(int(args.max_events), 0)]):
            if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                break
            frame = int(event.get("frame", 0))
            key = int(event.get("key", -1))
            interval_start, interval_end = active_interval(active_dense, frame, key)
            start = max(interval_start, frame - max(int(args.pre_dense_frames), 0))
            end = min(interval_end, frame + max(int(args.post_dense_frames), 0) + 1, dense.shape[0])
            if end <= start:
                continue
            previous = dense[start:end].copy()
            qpos = dense[min(max(frame, 0), dense.shape[0] - 1)]
            nearest = nearest_fingers_for_key(kin, qpos, key, int(args.top_fingers))
            best_score: dict[str, Any] | None = None
            best_slice: np.ndarray | None = None
            best_candidate: dict[str, Any] | None = None
            trials: list[dict[str, Any]] = []
            for item in nearest:
                finger = int(item["finger"])
                for mode in modes:
                    if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                        break
                    ref = reference_frame_for_mode(
                        mode=mode,
                        interval_start=interval_start,
                        interval_end=interval_end,
                        event_frame=frame,
                        substeps=substeps,
                        dense_total=dense.shape[0],
                    )
                    dense[start:end] = apply_finger_hold(
                        dense,
                        start=start,
                        end=end,
                        finger=finger,
                        reference_qpos=dense[ref],
                    )
                    eval_count += 1
                    trial, _ = simulate_score(
                        payload,
                        dense,
                        out / f"eval_{eval_count:04d}",
                        str(args.environment_name),
                        float(args.threshold),
                        dense_dt,
                    )
                    record = {
                        "finger": int(finger),
                        "mode": str(mode),
                        "reference_frame": int(ref),
                        "nearest": item,
                        "trial": trial,
                    }
                    trials.append(record)
                    if best_score is None or score_key(trial) > score_key(best_score):
                        best_score = dict(trial)
                        best_slice = dense[start:end].copy()
                        best_candidate = record
                    dense[start:end] = previous
                if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                    break

            delta = float(best_score["event_f1"] - current["event_f1"]) if best_score else 0.0
            accept = bool(best_score) and delta >= float(args.min_delta_event_f1)
            if (
                bool(best_score)
                and bool(args.allow_equal_f1_mispress_drop)
                and delta >= -1e-12
                and int(best_score["mispresses"]) < int(current["mispresses"])
            ):
                accept = True
            summary = {
                "event_index": int(event_index),
                "event": {"frame": int(frame), "key": int(key), "time_s": float(event.get("time_s", 0.0))},
                "interval_start": int(interval_start),
                "interval_end": int(interval_end),
                "dense_start": int(start),
                "dense_end": int(end),
                "nearest": nearest,
                "previous": current,
                "best_trial": best_score,
                "best_candidate": best_candidate,
                "delta_event_f1": float(delta),
                "trials": trials,
            }
            if accept and best_slice is not None and best_score is not None:
                dense[start:end] = best_slice
                held[start:end] = 1.0
                current = best_score
                accepted.append(summary)
            else:
                dense[start:end] = previous
                rejected.append(summary)

    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["planned_hand_joints"] = dense[::substeps][:control_steps].astype(np.float32)
    payload["planned_hand_velocities"] = compute_hand_velocities(payload["planned_hand_joints"], control_timestep=0.05)
    payload["legato_held_dense_frames"] = held
    atomic_save_npz(out / "trajectory.npz", **payload)
    final_score, _ = simulate_score(payload, dense, out / "rp1m_sim", str(args.environment_name), float(args.threshold), dense_dt)
    result = {
        "source_npz": str(args.source_npz),
        "output_dir": str(out),
        "pre_dense_frames": int(args.pre_dense_frames),
        "post_dense_frames": int(args.post_dense_frames),
        "top_fingers": int(args.top_fingers),
        "modes": modes,
        "duplicate_events": int(len(duplicate_events)),
        "initial": initial,
        "greedy_score": current,
        "final_score": final_score,
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "eval_count": int(eval_count),
        "accepted": accepted[:400],
        "rejected": rejected[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
