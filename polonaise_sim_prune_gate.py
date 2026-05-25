#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for _path in (
    REPO_ROOT,
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


@dataclass(frozen=True)
class Event:
    frame: int
    key: int


def _event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _events(binary: np.ndarray) -> list[Event]:
    rows = np.asarray(binary, dtype=bool)
    previous = np.zeros((rows.shape[1],), dtype=bool)
    out: list[Event] = []
    for frame, row in enumerate(rows):
        for key in np.flatnonzero(row & ~previous):
            out.append(Event(frame=int(frame), key=int(key)))
        previous = row
    return out


def _match(target_events: list[Event], played_events: list[Event], *, tol_frames: int) -> set[int]:
    by_key: dict[int, list[int]] = {}
    for index, event in enumerate(played_events):
        by_key.setdefault(int(event.key), []).append(index)
    used_played: set[int] = set()
    for target in target_events:
        best_index: int | None = None
        best_error = int(tol_frames) + 1
        for played_index in by_key.get(int(target.key), []):
            if played_index in used_played:
                continue
            error = abs(int(played_events[played_index].frame) - int(target.frame))
            if error <= int(tol_frames) and error < best_error:
                best_index = int(played_index)
                best_error = int(error)
        if best_index is not None:
            used_played.add(best_index)
    return used_played


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _intervals_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.float32).reshape(-1) > 0.5
    out: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(values.tolist()):
        if active and start is None:
            start = int(index)
        elif not active and start is not None:
            out.append((int(start), int(index)))
            start = None
    if start is not None:
        out.append((int(start), int(values.size)))
    return out


def _load_injection_intervals(candidate_npz: Path, candidate_summary: Path | None) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    if candidate_summary is not None and candidate_summary.exists():
        summary = json.loads(candidate_summary.read_text())
        for row in summary.get("injection_rows", []):
            start = int(row.get("dense_start", -1))
            end = int(row.get("dense_end", -1))
            if start >= 0 and end > start:
                intervals.append({"dense_start": start, "dense_end": end, "source": "summary", **row})
    if intervals:
        return intervals

    payload = _load_npz(candidate_npz)
    mask = None
    for key in ("siciliana_injected_dense_frames", "cadenza_injected_dense_frames"):
        if key in payload:
            mask = payload[key]
            break
    if mask is None:
        raise KeyError(
            "candidate trajectory does not contain an injection mask and no summary injection_rows were provided"
        )
    for start, end in _intervals_from_mask(mask):
        intervals.append({"dense_start": int(start), "dense_end": int(end), "source": "mask"})
    return intervals


def _simulate_and_score(
    *,
    payload: dict[str, np.ndarray],
    dense: np.ndarray,
    output_dir: Path,
    environment_name: str,
    dense_dt: float,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
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
    return score, goals, played


def _existing_rollout(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    rollout_path = path / "rp1m_sim" / "rollout.npz"
    if not rollout_path.exists():
        return None
    with np.load(rollout_path, allow_pickle=False) as rollout:
        return (
            np.asarray(rollout["goals"], dtype=np.float32),
            np.asarray(rollout["source_played_piano"], dtype=np.float32),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polonaise: prune injected hand-state pulses that simulator validation shows did not match target events."
    )
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-summary", default=None)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pre-window-frames", type=int, default=2)
    parser.add_argument("--post-window-frames", type=int, default=30)
    parser.add_argument("--max-unmatched-per-matched", type=float, default=999.0)
    parser.add_argument("--any-matched-event-keeps", action="store_true")
    parser.add_argument("--revert-empty-intervals", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_npz = Path(args.candidate_npz)
    source_npz = Path(args.source_npz)
    candidate_summary = Path(args.candidate_summary) if args.candidate_summary else candidate_npz.parent / "summary.json"

    payload = _load_npz(candidate_npz)
    source_payload = _load_npz(source_npz)
    dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32).copy()
    source_dense = np.asarray(source_payload["planned_hand_joints_dense"], dtype=np.float32)
    if source_dense.shape != dense.shape:
        raise ValueError(f"source dense shape {source_dense.shape} does not match candidate dense shape {dense.shape}")

    control = np.asarray(payload["planned_hand_joints"], dtype=np.float32)
    substeps = max(int(dense.shape[0] // max(control.shape[0], 1)), 1)
    dense_dt = 0.05 / float(substeps)

    existing = _existing_rollout(candidate_npz.parent)
    if existing is None:
        old_score, goals, played = _simulate_and_score(
            payload=payload,
            dense=dense,
            output_dir=out / "candidate_rp1m_sim",
            environment_name=str(args.environment_name),
            dense_dt=float(dense_dt),
            threshold=float(args.threshold),
        )
    else:
        goals, played = existing
        old_score = score_rollout(
            target_keys=goals,
            played_keys=played,
            dt=float(dense_dt),
            threshold=float(args.threshold),
            timing_tolerance_s=0.15,
        )

    tol_frames = int(round(0.15 / float(dense_dt)))
    target_events = _events(np.asarray(goals, dtype=np.float32)[:, :88] > float(args.threshold))
    played_events = _events(np.asarray(played, dtype=np.float32)[:, :88] > float(args.threshold))
    used_played = _match(target_events, played_events, tol_frames=tol_frames)
    intervals = _load_injection_intervals(candidate_npz, candidate_summary)

    pruned = []
    kept = []
    for interval in intervals:
        start = int(interval["dense_start"])
        end = int(interval["dense_end"])
        intended_keys: set[int] = set()
        if "target_key" in interval:
            intended_keys.add(int(interval["target_key"]))
        for key in interval.get("hits", []) or []:
            intended_keys.add(int(key))
        window_start = max(0, start - max(int(args.pre_window_frames), 0))
        window_end = end + max(int(args.post_window_frames), 0)
        event_indices = [
            index
            for index, event in enumerate(played_events)
            if int(window_start) <= int(event.frame) < int(window_end)
        ]
        if intended_keys and not bool(args.any_matched_event_keeps):
            credited_indices = [
                index for index in event_indices if int(played_events[index].key) in intended_keys
            ]
        else:
            credited_indices = event_indices
        matched_count = int(sum(1 for index in credited_indices if index in used_played))
        unmatched_count = int(len(event_indices) - matched_count)
        should_revert = False
        reason = ""
        if not event_indices and bool(args.revert_empty_intervals):
            should_revert = True
            reason = "empty"
        elif matched_count == 0 and unmatched_count > 0:
            should_revert = True
            reason = "unmatched_only"
        elif unmatched_count > float(args.max_unmatched_per_matched) * max(matched_count, 1):
            should_revert = True
            reason = "unmatched_ratio"
        record = {
            **interval,
            "window_start": int(window_start),
            "window_end": int(window_end),
            "matched_events": int(matched_count),
            "unmatched_events": int(unmatched_count),
            "intended_keys": sorted(int(key) for key in intended_keys),
            "reason": reason,
        }
        if should_revert:
            dense[start:end] = source_dense[start:end]
            pruned.append(record)
        else:
            kept.append(record)

    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["polonaise_pruned_dense_frames"] = np.zeros((dense.shape[0],), dtype=np.float32)
    for row in pruned:
        payload["polonaise_pruned_dense_frames"][int(row["dense_start"]) : int(row["dense_end"])] = 1.0
    atomic_save_npz(out / "trajectory.npz", **payload)

    score, _goals, _played = _simulate_and_score(
        payload=payload,
        dense=dense,
        output_dir=out / "rp1m_sim",
        environment_name=str(args.environment_name),
        dense_dt=float(dense_dt),
        threshold=float(args.threshold),
    )
    result = {
        "candidate_npz": str(candidate_npz),
        "source_npz": str(source_npz),
        "output_dir": str(out),
        "intervals": int(len(intervals)),
        "pruned_intervals": int(len(pruned)),
        "kept_intervals": int(len(kept)),
        "old_event_f1": float(_event_f1(old_score)),
        "old_frame_f1": float(old_score.get("frame_f1", 0.0)),
        "old_matched": int(old_score.get("matched_press_events", 0)),
        "old_target": int(old_score.get("target_press_events", 0)),
        "old_played": int(old_score.get("played_press_events", 0)),
        "old_mispresses": int(old_score.get("mispresses", 0)),
        "event_f1": float(_event_f1(score)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "played": int(score.get("played_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "delta_event_f1": float(_event_f1(score) - _event_f1(old_score)),
        "delta_frame_f1": float(score.get("frame_f1", 0.0) - old_score.get("frame_f1", 0.0)),
        "pre_window_frames": int(args.pre_window_frames),
        "post_window_frames": int(args.post_window_frames),
        "max_unmatched_per_matched": float(args.max_unmatched_per_matched),
        "any_matched_event_keeps": bool(args.any_matched_event_keeps),
        "revert_empty_intervals": bool(args.revert_empty_intervals),
        "pruned": pruned[:400],
        "kept": kept[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
