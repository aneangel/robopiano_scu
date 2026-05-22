from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PressEvent:
    frame: int
    key: int

    @property
    def time_s(self) -> float:
        raise AttributeError("Use frame * dt for timing.")


def frame_metrics(target_keys: np.ndarray, played_keys: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    target = np.asarray(target_keys, dtype=np.float32)[:, :88] > float(threshold)
    played = np.asarray(played_keys, dtype=np.float32)[:, :88] > float(threshold)
    steps = min(int(target.shape[0]), int(played.shape[0]))
    target = target[:steps]
    played = played[:steps]
    tp = int(np.logical_and(target, played).sum())
    fp = int(np.logical_and(~target, played).sum())
    fn = int(np.logical_and(target, ~played).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "frame_true_positives": float(tp),
        "frame_false_positives": float(fp),
        "frame_false_negatives": float(fn),
        "frame_precision": precision,
        "frame_recall": recall,
        "frame_f1": f1,
    }


def press_events_from_roll(roll: np.ndarray, *, threshold: float = 0.5) -> list[PressEvent]:
    active = np.asarray(roll, dtype=np.float32)[:, :88] > float(threshold)
    previous = np.zeros((active.shape[1],), dtype=bool)
    events: list[PressEvent] = []
    for frame, row in enumerate(active):
        onsets = np.flatnonzero(row & ~previous)
        events.extend(PressEvent(frame=int(frame), key=int(key)) for key in onsets.tolist())
        previous = row
    return events


def match_press_events(
    target_events: list[PressEvent],
    played_events: list[PressEvent],
    *,
    tolerance_frames: int,
) -> tuple[list[dict[str, int]], list[PressEvent], list[PressEvent]]:
    by_key: dict[int, list[int]] = {}
    for index, event in enumerate(played_events):
        by_key.setdefault(int(event.key), []).append(index)

    used: set[int] = set()
    matches: list[dict[str, int]] = []
    missed: list[PressEvent] = []
    for target in target_events:
        best_index = None
        best_abs = 10**9
        for played_index in by_key.get(int(target.key), []):
            if played_index in used:
                continue
            played = played_events[played_index]
            err = int(played.frame) - int(target.frame)
            abs_err = abs(err)
            if abs_err <= int(tolerance_frames) and abs_err < best_abs:
                best_index = played_index
                best_abs = abs_err
        if best_index is None:
            missed.append(target)
            continue
        used.add(best_index)
        played = played_events[best_index]
        matches.append(
            {
                "key": int(target.key),
                "target_frame": int(target.frame),
                "played_frame": int(played.frame),
                "signed_error_frames": int(played.frame - target.frame),
                "abs_error_frames": int(abs(played.frame - target.frame)),
            }
        )
    false_events = [event for index, event in enumerate(played_events) if index not in used]
    return matches, missed, false_events


def event_metrics(
    target_keys: np.ndarray,
    played_keys: np.ndarray,
    *,
    dt: float,
    threshold: float = 0.5,
    tolerance_s: float = 0.15,
) -> dict[str, Any]:
    tolerance_frames = int(np.ceil(float(tolerance_s) / max(float(dt), 1e-9)))
    target_events = press_events_from_roll(target_keys, threshold=threshold)
    played_events = press_events_from_roll(played_keys, threshold=threshold)
    matches, missed, false_events = match_press_events(target_events, played_events, tolerance_frames=tolerance_frames)
    matched = float(len(matches))
    precision = float(matched / max(matched + len(false_events), 1.0))
    recall = float(matched / max(matched + len(missed), 1.0))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    errors = np.asarray([row["abs_error_frames"] for row in matches], dtype=np.float32) * float(dt)
    return {
        "target_press_events": int(len(target_events)),
        "played_press_events": int(len(played_events)),
        "matched_press_events": int(len(matches)),
        "missed_key_presses": int(len(missed)),
        "mispresses": int(len(false_events)),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "timing_abs_error_mean_s": float(errors.mean()) if errors.size else None,
        "timing_abs_error_p95_s": float(np.percentile(errors, 95)) if errors.size else None,
        "matches": matches[:200],
        "missed_events": [asdict(event) for event in missed[:200]],
        "mispress_events": [asdict(event) for event in false_events[:200]],
    }


def evaluate_rollout(
    target_keys: np.ndarray,
    played_keys: np.ndarray,
    *,
    dt: float,
    threshold: float = 0.5,
    tolerance_s: float = 0.15,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scored_steps": int(min(np.asarray(target_keys).shape[0], np.asarray(played_keys).shape[0])),
        "scored_keys": 88,
        "dt": float(dt),
        "threshold": float(threshold),
    }
    out.update(frame_metrics(target_keys, played_keys, threshold=threshold))
    out.update(event_metrics(target_keys, played_keys, dt=dt, threshold=threshold, tolerance_s=tolerance_s))
    return out


def evaluate_demo_baselines(demos: Any, *, dt: float, threshold: float = 0.5) -> dict[str, Any]:
    rows = []
    for row_index, demo_id in enumerate(np.asarray(demos.demo_ids).reshape(-1).tolist()):
        metrics = evaluate_rollout(demos.goals[row_index], demos.piano_states[row_index], dt=dt, threshold=threshold)
        rows.append({"demo_id": int(demo_id), **{key: value for key, value in metrics.items() if not isinstance(value, list)}})
    best = max(rows, key=lambda row: (row["event_f1"], row["frame_f1"])) if rows else None
    return {
        "num_demos": int(len(rows)),
        "best_demo": best,
        "mean_frame_f1": float(np.mean([row["frame_f1"] for row in rows])) if rows else 0.0,
        "mean_event_f1": float(np.mean([row["event_f1"] for row in rows])) if rows else 0.0,
        "demos": rows,
    }
