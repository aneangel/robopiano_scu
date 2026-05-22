from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

import numpy as np

from nocturne.offline_eval import PressEvent, match_press_events, press_events_from_roll


def analyze_demo_error_consensus(
    demos: Any,
    *,
    song_name: str,
    dt: float,
    threshold: float = 0.5,
    tolerance_s: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Find target key presses and false key presses that are common across demos."""
    tolerance_frames = int(np.ceil(float(tolerance_s) / max(float(dt), 1e-9)))
    target_events = press_events_from_roll(demos.goals[0], threshold=threshold)
    target_index = {(int(event.frame), int(event.key)): index for index, event in enumerate(target_events)}
    target_frames = np.asarray([event.frame for event in target_events], dtype=np.int64)
    target_keys_by_frame = _target_keys_by_frame(target_events)

    hit_counts: dict[tuple[int, int], int] = defaultdict(int)
    miss_counts: dict[tuple[int, int], int] = defaultdict(int)
    timing_errors: dict[tuple[int, int], list[int]] = defaultdict(list)
    false_buckets: dict[tuple[int, int], dict[str, Any]] = {}

    for row_index, demo_id in enumerate(np.asarray(demos.demo_ids).reshape(-1).tolist()):
        played_events = press_events_from_roll(demos.piano_states[row_index], threshold=threshold)
        matches, missed, false_events = match_press_events(
            target_events,
            played_events,
            tolerance_frames=tolerance_frames,
        )
        for match in matches:
            key = (int(match["target_frame"]), int(match["key"]))
            hit_counts[key] += 1
            timing_errors[key].append(int(match["signed_error_frames"]))
        for event in missed:
            miss_counts[(int(event.frame), int(event.key))] += 1
        for event in false_events:
            bucket = _false_event_bucket(
                event,
                target_frames=target_frames,
                target_keys_by_frame=target_keys_by_frame,
                tolerance_frames=tolerance_frames,
            )
            row = false_buckets.setdefault(
                (int(bucket["context_frame"]), int(event.key)),
                {
                    "song_name": song_name,
                    "context_frame": int(bucket["context_frame"]),
                    "context_time_s": float(bucket["context_frame"] * dt),
                    "context_type": bucket["context_type"],
                    "played_key": int(event.key),
                    "nearest_target_frame": bucket["nearest_target_frame"],
                    "nearest_target_time_s": (
                        None if bucket["nearest_target_frame"] is None else float(bucket["nearest_target_frame"] * dt)
                    ),
                    "nearest_target_keys": bucket["nearest_target_keys"],
                    "demo_ids": [],
                    "played_frames": [],
                },
            )
            row["demo_ids"].append(int(demo_id))
            row["played_frames"].append(int(event.frame))

    num_demos = int(demos.num_demos)
    missed_rows = []
    for event in target_events:
        key = (int(event.frame), int(event.key))
        errors = np.asarray(timing_errors.get(key, []), dtype=np.float32)
        hits = int(hit_counts.get(key, 0))
        misses = int(miss_counts.get(key, num_demos - hits))
        missed_rows.append(
            {
                "song_name": song_name,
                "event_index": int(target_index[key]),
                "target_frame": int(event.frame),
                "target_time_s": float(event.frame * dt),
                "target_key": int(event.key),
                "num_demos": num_demos,
                "hit_demo_count": hits,
                "missed_demo_count": misses,
                "miss_rate": float(misses / max(num_demos, 1)),
                "missed_by_all_demos": bool(misses == num_demos),
                "timing_error_mean_frames": float(errors.mean()) if errors.size else None,
                "timing_error_abs_p95_frames": float(np.percentile(np.abs(errors), 95)) if errors.size else None,
            }
        )

    mispress_rows = []
    for row in false_buckets.values():
        demo_ids = sorted(set(int(value) for value in row["demo_ids"]))
        frames = np.asarray(row["played_frames"], dtype=np.float32)
        mispress_rows.append(
            {
                **{key: value for key, value in row.items() if key not in {"demo_ids", "played_frames"}},
                "num_demos": num_demos,
                "demo_count": int(len(demo_ids)),
                "demo_rate": float(len(demo_ids) / max(num_demos, 1)),
                "seen_in_all_demos": bool(len(demo_ids) == num_demos),
                "demo_ids": ",".join(str(value) for value in demo_ids),
                "played_frame_mean": float(frames.mean()) if frames.size else None,
                "played_frame_min": int(frames.min()) if frames.size else None,
                "played_frame_max": int(frames.max()) if frames.size else None,
            }
        )

    missed_rows.sort(key=lambda row: (-float(row["miss_rate"]), int(row["target_frame"]), int(row["target_key"])))
    mispress_rows.sort(key=lambda row: (-float(row["demo_rate"]), int(row["context_frame"]), int(row["played_key"])))
    return {"missed_target_events": missed_rows, "mispress_buckets": mispress_rows}


def analyze_stitched_error_context(
    demos: Any,
    *,
    song_name: str,
    stitched_goals: np.ndarray,
    stitched_piano_states: np.ndarray,
    dt: float,
    threshold: float = 0.5,
    tolerance_s: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Classify stitched errors by whether any raw demo contained a usable local moment."""
    consensus = analyze_demo_error_consensus(
        demos,
        song_name=song_name,
        dt=dt,
        threshold=threshold,
        tolerance_s=tolerance_s,
    )
    target_lookup = {
        (int(row["target_frame"]), int(row["target_key"])): row for row in consensus["missed_target_events"]
    }
    mispress_lookup = {
        (int(row["context_frame"]), int(row["played_key"])): row for row in consensus["mispress_buckets"]
    }

    tolerance_frames = int(np.ceil(float(tolerance_s) / max(float(dt), 1e-9)))
    target_events = press_events_from_roll(stitched_goals, threshold=threshold)
    played_events = press_events_from_roll(stitched_piano_states, threshold=threshold)
    matches, missed, false_events = match_press_events(
        target_events,
        played_events,
        tolerance_frames=tolerance_frames,
    )
    matched_lookup = {(int(row["target_frame"]), int(row["key"])): row for row in matches}
    target_frames = np.asarray([event.frame for event in target_events], dtype=np.int64)
    target_keys_by_frame = _target_keys_by_frame(target_events)

    missed_rows = []
    for event in missed:
        key = (int(event.frame), int(event.key))
        raw = target_lookup.get(key, {})
        raw_hits = int(raw.get("hit_demo_count", 0))
        missed_rows.append(
            {
                "song_name": song_name,
                "target_frame": int(event.frame),
                "target_time_s": float(event.frame * dt),
                "target_key": int(event.key),
                "raw_hit_demo_count": raw_hits,
                "raw_missed_demo_count": int(raw.get("missed_demo_count", demos.num_demos)),
                "raw_miss_rate": float(raw.get("miss_rate", 1.0)),
                "avoidable_from_raw_demos": bool(raw_hits > 0),
                "missed_by_all_raw_demos": bool(raw_hits == 0),
            }
        )

    mispress_rows = []
    for event in false_events:
        bucket = _false_event_bucket(
            event,
            target_frames=target_frames,
            target_keys_by_frame=target_keys_by_frame,
            tolerance_frames=tolerance_frames,
        )
        raw = mispress_lookup.get((int(bucket["context_frame"]), int(event.key)), {})
        raw_seen = int(raw.get("demo_count", 0))
        mispress_rows.append(
            {
                "song_name": song_name,
                "played_frame": int(event.frame),
                "played_time_s": float(event.frame * dt),
                "played_key": int(event.key),
                "context_frame": int(bucket["context_frame"]),
                "context_type": bucket["context_type"],
                "nearest_target_frame": bucket["nearest_target_frame"],
                "nearest_target_keys": bucket["nearest_target_keys"],
                "raw_mispress_demo_count": raw_seen,
                "raw_mispress_demo_rate": float(raw.get("demo_rate", 0.0)),
                "seen_in_raw_demos": bool(raw_seen > 0),
            }
        )

    matched_rows = []
    for event in target_events:
        key = (int(event.frame), int(event.key))
        if key not in matched_lookup:
            continue
        match = matched_lookup[key]
        raw = target_lookup.get(key, {})
        matched_rows.append(
            {
                "song_name": song_name,
                "target_frame": int(event.frame),
                "target_time_s": float(event.frame * dt),
                "target_key": int(event.key),
                "played_frame": int(match["played_frame"]),
                "signed_error_frames": int(match["signed_error_frames"]),
                "raw_hit_demo_count": int(raw.get("hit_demo_count", 0)),
                "raw_miss_rate": float(raw.get("miss_rate", 1.0)),
            }
        )

    return {"stitched_missed_events": missed_rows, "stitched_mispress_events": mispress_rows, "stitched_matches": matched_rows}


def _target_keys_by_frame(events: list[PressEvent]) -> dict[int, str]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for event in events:
        by_frame[int(event.frame)].append(int(event.key))
    return {frame: ",".join(str(key) for key in sorted(keys)) for frame, keys in by_frame.items()}


def _false_event_bucket(
    event: PressEvent,
    *,
    target_frames: np.ndarray,
    target_keys_by_frame: dict[int, str],
    tolerance_frames: int,
) -> dict[str, Any]:
    if target_frames.size == 0:
        return {
            "context_frame": int(event.frame),
            "context_type": "no_targets",
            "nearest_target_frame": None,
            "nearest_target_keys": "",
        }
    nearest = _nearest_frame(int(event.frame), target_frames)
    context_radius = max(int(tolerance_frames) * 2, 1)
    if abs(int(event.frame) - int(nearest)) <= context_radius:
        return {
            "context_frame": int(nearest),
            "context_type": "near_target",
            "nearest_target_frame": int(nearest),
            "nearest_target_keys": target_keys_by_frame.get(int(nearest), ""),
        }
    bucket_width = max(context_radius, 1)
    context_frame = int(round(float(event.frame) / float(bucket_width)) * bucket_width)
    return {
        "context_frame": context_frame,
        "context_type": "between_targets",
        "nearest_target_frame": int(nearest),
        "nearest_target_keys": target_keys_by_frame.get(int(nearest), ""),
    }


def _nearest_frame(frame: int, sorted_frames: np.ndarray) -> int:
    index = int(np.searchsorted(sorted_frames, int(frame)))
    candidates = []
    if index < sorted_frames.size:
        candidates.append(int(sorted_frames[index]))
    if index > 0:
        candidates.append(int(sorted_frames[index - 1]))
    return min(candidates, key=lambda value: abs(value - int(frame)))
