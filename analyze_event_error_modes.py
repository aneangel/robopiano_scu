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
for _path in (REPO_ROOT / "Intermezzo" / "src", REPO_ROOT / "partita" / "src"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intermezzo.io import atomic_save_json  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402


@dataclass(frozen=True)
class Event:
    frame: int
    key: int


def _events(binary: np.ndarray) -> list[Event]:
    rows = np.asarray(binary, dtype=bool)
    previous = np.zeros((rows.shape[1],), dtype=bool)
    out: list[Event] = []
    for frame, row in enumerate(rows):
        for key in np.flatnonzero(row & ~previous):
            out.append(Event(frame=int(frame), key=int(key)))
        previous = row
    return out


def _match(target_events: list[Event], played_events: list[Event], *, tol_frames: int) -> tuple[set[int], set[int]]:
    by_key: dict[int, list[int]] = {}
    for index, event in enumerate(played_events):
        by_key.setdefault(event.key, []).append(index)
    used_played: set[int] = set()
    used_target: set[int] = set()
    for target_index, target in enumerate(target_events):
        best_index: int | None = None
        best_error = tol_frames + 1
        for played_index in by_key.get(target.key, []):
            if played_index in used_played:
                continue
            error = abs(int(played_events[played_index].frame) - int(target.frame))
            if error <= tol_frames and error < best_error:
                best_index = played_index
                best_error = error
        if best_index is not None:
            used_played.add(best_index)
            used_target.add(target_index)
    return used_target, used_played


def _rollout_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "goals" in data.files:
            target = np.asarray(data["goals"], dtype=np.float32)[:, :88]
        elif "target_keys" in data.files:
            target = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
        else:
            raise KeyError(f"{path} does not contain goals or target_keys")
        if "source_played_piano" in data.files:
            played = np.asarray(data["source_played_piano"], dtype=np.float32)[:, :88]
        elif "played_keys" in data.files:
            played = np.asarray(data["played_keys"], dtype=np.float32)[:, :88]
        else:
            raise KeyError(f"{path} does not contain source_played_piano or played_keys")
    steps = min(target.shape[0], played.shape[0])
    return target[:steps], played[:steps]


def analyze_one(run_dir: Path, *, threshold: float, dt: float, tolerance_s: float) -> dict[str, Any]:
    rollout_path = run_dir / "rp1m_sim" / "rollout.npz"
    if not rollout_path.exists():
        rollout_path = run_dir / "rollout.npz"
    target_raw, played_raw = _rollout_arrays(rollout_path)
    target = target_raw > float(threshold)
    played = played_raw > float(threshold)
    target_events = _events(target)
    played_events = _events(played)
    tol_frames = int(round(float(tolerance_s) / float(dt)))
    used_target, used_played = _match(target_events, played_events, tol_frames=tol_frames)

    duplicate_active = 0
    wrong_during_active = 0
    wrong_during_silence = 0
    same_key_near = 0
    closest_errors: list[int] = []
    top_unused_keys: dict[int, int] = {}
    for index, event in enumerate(played_events):
        if index in used_played:
            continue
        frame = min(max(int(event.frame), 0), target.shape[0] - 1)
        key = int(event.key)
        top_unused_keys[key] = top_unused_keys.get(key, 0) + 1
        if target[frame, key]:
            duplicate_active += 1
        elif np.any(target[frame]):
            wrong_during_active += 1
        else:
            wrong_during_silence += 1
        same_key_targets = [target_event.frame for target_event in target_events if target_event.key == key]
        if same_key_targets:
            closest = min(abs(frame - int(target_frame)) for target_frame in same_key_targets)
            closest_errors.append(int(closest))
            if closest <= tol_frames:
                same_key_near += 1

    score = score_rollout(
        target_keys=target_raw,
        played_keys=played_raw,
        dt=float(dt),
        threshold=float(threshold),
        timing_tolerance_s=float(tolerance_s),
    )
    matched = int(score.get("matched_press_events", len(used_played)))
    played_count = int(score.get("played_press_events", len(played_events)))
    target_count = int(score.get("target_press_events", len(target_events)))
    precision = matched / played_count if played_count else 0.0
    recall = matched / target_count if target_count else 0.0
    return {
        "run_dir": str(run_dir),
        "matched": matched,
        "target": target_count,
        "played": played_count,
        "missed": int(target_count - matched),
        "mispresses": int(played_count - matched),
        "precision": float(precision),
        "recall": float(recall),
        "event_f1": float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "duplicate_active_key_restrikes": int(duplicate_active),
        "wrong_key_while_target_active": int(wrong_during_active),
        "wrong_key_while_target_silent": int(wrong_during_silence),
        "unused_same_key_near_target_onset": int(same_key_near),
        "unused_same_key_closest_frame_median": float(np.median(closest_errors)) if closest_errors else None,
        "top_unused_keys": [
            {"key": int(key), "events": int(count)}
            for key, count in sorted(top_unused_keys.items(), key=lambda item: (-item[1], item[0]))[:12]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify Impromptu event-level mispress modes.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--timing-tolerance-s", type=float, default=0.15)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for run_dir in sorted(root.glob("maestro_*")):
        if run_dir.is_dir():
            rows.append(
                analyze_one(
                    run_dir,
                    threshold=float(args.threshold),
                    dt=float(args.dt),
                    tolerance_s=float(args.timing_tolerance_s),
                )
            )
    totals = {
        "root": str(root),
        "runs": rows,
        "runs_completed": int(len(rows)),
        "matched": int(sum(row["matched"] for row in rows)),
        "target": int(sum(row["target"] for row in rows)),
        "played": int(sum(row["played"] for row in rows)),
        "missed": int(sum(row["missed"] for row in rows)),
        "mispresses": int(sum(row["mispresses"] for row in rows)),
        "duplicate_active_key_restrikes": int(sum(row["duplicate_active_key_restrikes"] for row in rows)),
        "wrong_key_while_target_active": int(sum(row["wrong_key_while_target_active"] for row in rows)),
        "wrong_key_while_target_silent": int(sum(row["wrong_key_while_target_silent"] for row in rows)),
        "unused_same_key_near_target_onset": int(sum(row["unused_same_key_near_target_onset"] for row in rows)),
        "mean_event_f1": float(np.mean([row["event_f1"] for row in rows])) if rows else 0.0,
        "mean_frame_f1": float(np.mean([row["frame_f1"] for row in rows])) if rows else 0.0,
    }
    atomic_save_json(Path(args.output_json), totals)
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
