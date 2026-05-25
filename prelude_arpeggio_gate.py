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

from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.planner import plan_target_keys  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def intervals_for_key(active: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(active, dtype=bool).reshape(-1)
    intervals = []
    frame = 0
    while frame < values.size:
        if not values[frame]:
            frame += 1
            continue
        start = frame
        while frame + 1 < values.size and values[frame + 1]:
            frame += 1
        intervals.append((int(start), int(frame)))
        frame += 1
    return intervals


def hand_ordered_chunks(keys: np.ndarray, *, split_key: int, max_keys_per_frame: int) -> list[np.ndarray]:
    values = np.asarray(keys, dtype=np.int32).reshape(-1)
    left = np.sort(values[values < int(split_key)])
    right = np.sort(values[values >= int(split_key)])
    ordered: list[int] = []
    while left.size or right.size:
        if left.size:
            ordered.append(int(left[0]))
            left = left[1:]
        if right.size:
            ordered.append(int(right[0]))
            right = right[1:]
    width = max(int(max_keys_per_frame), 1)
    return [np.asarray(ordered[index : index + width], dtype=np.int32) for index in range(0, len(ordered), width)]


def arpeggiate_target_keys(
    target_keys: np.ndarray,
    *,
    threshold: float,
    split_key: int,
    max_keys_per_frame: int,
    max_shift_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    original = np.asarray(target_keys, dtype=np.float32)[:, :88]
    active = original > float(threshold)
    shifted = np.zeros_like(active, dtype=bool)
    shift_by_start_key: dict[tuple[int, int], int] = {}
    previous = np.zeros((88,), dtype=bool)
    arpeggiated_onsets = 0
    max_onset_polyphony = 0
    for frame, row in enumerate(active):
        onset_keys = np.flatnonzero(row & ~previous).astype(np.int32)
        max_onset_polyphony = max(max_onset_polyphony, int(onset_keys.size))
        if onset_keys.size > max(int(max_keys_per_frame), 1):
            arpeggiated_onsets += int(onset_keys.size)
            chunks = hand_ordered_chunks(
                onset_keys,
                split_key=int(split_key),
                max_keys_per_frame=max(int(max_keys_per_frame), 1),
            )
            for chunk_index, chunk in enumerate(chunks):
                shift = min(int(chunk_index), max(int(max_shift_frames), 0))
                for key in chunk:
                    shift_by_start_key[(int(frame), int(key))] = int(shift)
        previous = row

    for key in range(88):
        for start, end in intervals_for_key(active[:, key]):
            shift = int(shift_by_start_key.get((int(start), int(key)), 0))
            shifted_start = min(int(start) + shift, int(end))
            shifted[shifted_start : int(end) + 1, key] = True

    planned = np.asarray(target_keys, dtype=np.float32).copy()
    planned[:, :88] = shifted.astype(np.float32)
    active_counts = shifted.sum(axis=1).astype(np.int32)
    return planned.astype(np.float32), {
        "arpeggiated_onset_keys": int(arpeggiated_onsets),
        "max_original_onset_polyphony": int(max_onset_polyphony),
        "max_planned_polyphony": int(active_counts.max()) if active_counts.size else 0,
        "mean_planned_polyphony": float(active_counts.mean()) if active_counts.size else 0.0,
        "max_keys_per_frame": int(max_keys_per_frame),
        "max_shift_frames": int(max_shift_frames),
    }


def score_trajectory(
    *,
    trajectory_payload: dict[str, np.ndarray],
    original_target_keys: np.ndarray,
    out: Path,
    environment_name: str,
    threshold: float,
) -> dict[str, Any]:
    control = np.asarray(trajectory_payload["planned_hand_joints"], dtype=np.float32)
    dense = np.asarray(trajectory_payload["planned_hand_joints_dense"], dtype=np.float32)
    control_steps = int(original_target_keys.shape[0])
    substeps = max(int(dense.shape[0] // max(control_steps, 1)), 1)
    dense_goals = np.repeat(np.asarray(original_target_keys, dtype=np.float32)[:, :88], substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    payload = {key: np.asarray(value) for key, value in trajectory_payload.items()}
    payload["target_keys"] = np.asarray(original_target_keys, dtype=np.float32)
    atomic_save_npz(out / "trajectory.npz", **payload)
    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(environment_name),
        demo_id=0,
        actions=np.zeros((dense.shape[0], 39), dtype=np.float32),
        goals=dense_goals,
        hand_joints=dense,
        environment_name=str(environment_name),
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
            threshold=float(threshold),
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
        threshold=float(threshold),
        timing_tolerance_s=0.15,
    )
    return {
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prelude: arpeggiate high-polyphony onset masks for planning only.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-key", type=int, default=44)
    parser.add_argument("--max-keys-per-frame", type=int, nargs="+", default=[2])
    parser.add_argument("--max-shift-frames", type=int, nargs="+", default=[2, 3])
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        source_payload = {key: np.asarray(data[key]) for key in data.files}
    original = np.asarray(source_payload["target_keys"], dtype=np.float32)
    rows = []
    for max_keys in args.max_keys_per_frame:
        for max_shift in args.max_shift_frames:
            planned_keys, arp_meta = arpeggiate_target_keys(
                original,
                threshold=float(args.threshold),
                split_key=int(args.split_key),
                max_keys_per_frame=int(max_keys),
                max_shift_frames=int(max_shift),
            )
            config = ImpromptuConfig(
                environment_name=str(args.environment_name),
                threshold=float(args.threshold),
                control_timestep=0.05,
                interpolation_substeps=10,
                trajectory_mode="joint_space_straighten",
                key_press_depth=0.006,
                assignment_strategy="ik_aware_topk",
                assignment_top_k=8,
                ik_static_contact_validation=True,
                ik_multistart_seed_count=4,
                ik_static_contact_wrong_key_weight=0.25,
                ik_static_contact_missed_key_weight=4.0,
                ik_static_contact_failure_weight=25.0,
            )
            variant = out / f"max{int(max_keys)}_shift{int(max_shift)}"
            variant.mkdir(parents=True, exist_ok=True)
            trajectory = plan_target_keys(planned_keys, config=config)
            payload = trajectory.npz_payload()
            payload["prelude_planning_target_keys"] = planned_keys.astype(np.float32)
            payload["target_keys"] = original.astype(np.float32)
            score = score_trajectory(
                trajectory_payload=payload,
                original_target_keys=original,
                out=variant,
                environment_name=str(args.environment_name),
                threshold=float(args.threshold),
            )
            metadata = {
                "source_npz": str(source),
                "output_dir": str(variant),
                "max_keys_per_frame": int(max_keys),
                "max_shift_frames": int(max_shift),
                **arp_meta,
                **score,
            }
            atomic_save_json(variant / "summary.json", metadata)
            rows.append(metadata)
            print(json.dumps(metadata, sort_keys=True), flush=True)
    best = max(rows, key=lambda row: (float(row["event_f1"]), float(row["frame_f1"]))) if rows else None
    summary = {"source_npz": str(source), "output_dir": str(out), "rows": rows, "best": best}
    atomic_save_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
