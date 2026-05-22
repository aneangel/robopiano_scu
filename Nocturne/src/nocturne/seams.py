from __future__ import annotations

import json
from typing import Any

import numpy as np

from nocturne.events import event_key_mask, press_frame_indices, protected_press_mask
from nocturne.schema import NoteEvent, SegmentCandidate, StitchConfig


STITCH_ARRAYS = ("goals", "actions", "hand_joints", "hand_fingertips", "piano_states")


def stitch_selected(
    demos: Any,
    selected: list[SegmentCandidate],
    events: list[NoteEvent],
    intervals: list[tuple[int, int]],
    transition_costs: list[float],
    *,
    config: StitchConfig,
) -> dict[str, np.ndarray]:
    if not selected:
        raise ValueError("Cannot stitch an empty selected path.")
    num_frames = int(demos.num_frames)
    payload: dict[str, np.ndarray] = {}
    for name in STITCH_ARRAYS:
        source = demos.array(name)
        payload[name] = np.zeros((num_frames, source.shape[-1]), dtype=source.dtype)
    source_demo = np.full((num_frames,), -1, dtype=np.int64)
    segment_ids = np.full((num_frames,), -1, dtype=np.int64)
    local_scores = np.asarray([candidate.local_score for candidate in selected], dtype=np.float32)
    for candidate, (start, end) in zip(selected, intervals):
        row = _demo_row(demos, candidate.demo_id)
        for name in STITCH_ARRAYS:
            payload[name][start:end] = demos.array(name)[row, start:end]
        source_demo[start:end] = int(candidate.demo_id)
        segment_ids[start:end] = int(candidate.segment_id)
    payload["source_demo_id_per_frame"] = source_demo
    payload["segment_id_per_frame"] = segment_ids
    payload["local_score_per_segment"] = local_scores
    payload["transition_cost_per_boundary"] = np.asarray(transition_costs, dtype=np.float32)
    payload["stitch_interval_starts"] = np.asarray([int(start) for start, _ in intervals], dtype=np.int64)
    payload["stitch_interval_ends"] = np.asarray([int(end) for _, end in intervals], dtype=np.int64)
    payload["seam_frame_indices"] = np.asarray(seam_frames_from_intervals(intervals), dtype=np.int64)
    payload["event_indices"] = press_frame_indices(events)
    payload["press_frame_indices"] = press_frame_indices(events)
    payload["event_keys"] = event_key_mask(events)
    payload["dt"] = np.asarray(float(config.dt), dtype=np.float32)
    payload["metadata_json"] = np.asarray(
        json.dumps(
            {
                "module": "Nocturne",
                "num_events": len(events),
                "config": {
                    "dt": config.dt,
                    "window_frames": config.window_frames,
                    "pre_frames": config.pre_frames,
                    "event_tolerance_frames": config.event_tolerance_frames,
                    "seam_blend_radius": config.seam_blend_radius,
                    "objective_mode": config.objective_mode,
                    "repair_enabled": config.repair_enabled,
                    "repair_max_passes": config.repair_max_passes,
                    "repair_transition_margin": config.repair_transition_margin,
                    "adaptive_seam_enabled": config.adaptive_seam_enabled,
                    "seam_search_margin_frames": config.seam_search_margin_frames,
                    "transition_interpolation_enabled": config.transition_interpolation_enabled,
                },
            },
            sort_keys=True,
        )
    )
    return payload


def adaptive_min_distance_intervals(
    demos: Any,
    selected: list[SegmentCandidate],
    events: list[NoteEvent],
    intervals: list[tuple[int, int]],
    *,
    config: StitchConfig,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    if not bool(config.adaptive_seam_enabled):
        return intervals, {"status": "disabled", "num_boundaries": max(len(intervals) - 1, 0), "num_changed": 0}
    if len(selected) <= 1:
        return intervals, {"status": "skipped", "reason": "single_segment", "num_boundaries": 0, "num_changed": 0}

    margin = max(int(config.seam_search_margin_frames), 0)
    default_boundaries = seam_frames_from_intervals(intervals)
    boundaries: list[int] = []
    rows: list[dict[str, Any]] = []
    for index, default in enumerate(default_boundaries):
        previous = selected[index]
        current = selected[index + 1]
        left_limit = max(int(events[index].onset_frame) + margin, int(intervals[index][0]) + 1)
        right_limit = min(int(events[index + 1].onset_frame) - margin, int(intervals[index + 1][1]) - 1)
        if right_limit < left_limit:
            seam = int(default)
            score = _boundary_distance(demos, previous, current, seam)
            status = "fallback_empty_search"
        else:
            best = min(
                ((frame, _boundary_distance(demos, previous, current, frame)) for frame in range(left_limit, right_limit + 1)),
                key=lambda item: (float(item[1]), abs(int(item[0]) - int(default))),
            )
            seam, score = int(best[0]), float(best[1])
            status = "ok"
        boundaries.append(seam)
        rows.append(
            {
                "boundary_index": int(index),
                "previous_event_index": int(previous.event_index),
                "current_event_index": int(current.event_index),
                "previous_demo_id": int(previous.demo_id),
                "current_demo_id": int(current.demo_id),
                "default_seam_frame": int(default),
                "selected_seam_frame": int(seam),
                "search_start_frame": int(left_limit),
                "search_end_frame": int(right_limit),
                "hand_distance": float(score),
                "status": status,
            }
        )
    adjusted = _intervals_from_boundaries(boundaries, int(demos.num_frames))
    return adjusted, {
        "status": "ok",
        "num_boundaries": int(len(boundaries)),
        "num_changed": int(sum(int(row["default_seam_frame"]) != int(row["selected_seam_frame"]) for row in rows)),
        "mean_abs_shift_frames": float(np.mean([abs(int(row["selected_seam_frame"]) - int(row["default_seam_frame"])) for row in rows])) if rows else 0.0,
        "boundaries": rows,
    }


def smooth_stitched_payload(
    payload: dict[str, np.ndarray],
    *,
    seam_frames: list[int],
    press_frames: np.ndarray,
    blend_radius: int,
    threshold: float = 0.5,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    out = {key: np.array(value, copy=True) for key, value in payload.items()}
    radius = max(int(blend_radius), 0)
    if radius <= 0:
        return out
    protected = protected_press_mask(out["goals"], press_frames, radius=2, threshold=threshold)
    for name in ("hand_joints", "hand_fingertips", "actions"):
        if name not in out:
            continue
        arr = np.asarray(out[name], dtype=np.float32).copy()
        for seam in seam_frames:
            _blend_around_seam(arr, int(seam), radius=radius, protected=protected)
        out[name] = arr.astype(payload[name].dtype, copy=False)
    if joint_lower is not None and joint_upper is not None and "hand_joints" in out:
        out["hand_joints"] = np.clip(
            np.asarray(out["hand_joints"], dtype=np.float32),
            np.asarray(joint_lower, dtype=np.float32).reshape(1, -1),
            np.asarray(joint_upper, dtype=np.float32).reshape(1, -1),
        ).astype(np.float32)
    return out


def seam_frames_from_intervals(intervals: list[tuple[int, int]]) -> list[int]:
    return [int(end) for _, end in intervals[:-1]]


def joint_bounds_from_demos(demos: Any) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(demos.hand_joints, dtype=np.float32).reshape(-1, demos.hand_joints.shape[-1])
    return np.nanmin(joints, axis=0).astype(np.float32), np.nanmax(joints, axis=0).astype(np.float32)


def seam_jump_metrics(payload: dict[str, np.ndarray], seam_frames: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("hand_joints", "hand_fingertips", "actions"):
        if name not in payload:
            continue
        arr = np.asarray(payload[name], dtype=np.float32)
        jumps = []
        for seam in seam_frames:
            if 0 < int(seam) < arr.shape[0]:
                jumps.append(float(np.linalg.norm(arr[int(seam)] - arr[int(seam) - 1])))
        values = np.asarray(jumps, dtype=np.float32)
        out[f"seam/{name}_jump_mean"] = float(values.mean()) if values.size else 0.0
        out[f"seam/{name}_jump_p95"] = float(np.percentile(values, 95)) if values.size else 0.0
        adjacent = np.linalg.norm(np.diff(arr, axis=0).reshape(max(arr.shape[0] - 1, 0), -1), axis=1) if arr.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
        out[f"seam/{name}_within_demo_adjacent_p95_proxy"] = float(np.percentile(adjacent, 95)) if adjacent.size else 0.0
    return out


def _blend_around_seam(arr: np.ndarray, seam: int, *, radius: int, protected: np.ndarray) -> None:
    total = int(arr.shape[0])
    left = max(int(seam) - int(radius), 0)
    right = min(int(seam) + int(radius), total - 1)
    if right <= left + 1:
        return
    start_value = arr[left].copy()
    end_value = arr[right].copy()
    span = float(right - left)
    for frame in range(left + 1, right):
        if bool(protected[frame]):
            continue
        alpha = (float(frame - left) / span)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        arr[frame] = (1.0 - smooth) * start_value + smooth * end_value


def _intervals_from_boundaries(boundaries: list[int], num_frames: int) -> list[tuple[int, int]]:
    clipped = [int(np.clip(boundary, 1, max(int(num_frames) - 1, 1))) for boundary in boundaries]
    clipped = sorted(clipped)
    starts = [0, *clipped]
    ends = [*clipped, int(num_frames)]
    return [(int(start), int(max(end, start + 1))) for start, end in zip(starts, ends)]


def _boundary_distance(demos: Any, previous: SegmentCandidate, current: SegmentCandidate, seam: int) -> float:
    seam = int(np.clip(seam, 1, int(demos.num_frames) - 1))
    prev_row = _demo_row(demos, previous.demo_id)
    curr_row = _demo_row(demos, current.demo_id)
    left = seam - 1
    right = seam
    joint = _mean_l2(demos.hand_joints[prev_row, left], demos.hand_joints[curr_row, right])
    tips = _mean_l2(demos.hand_fingertips[prev_row, left], demos.hand_fingertips[curr_row, right])
    action = _mean_l2(demos.actions[prev_row, left], demos.actions[curr_row, right])
    return float(joint + 2.0 * tips + 0.25 * action)


def _mean_l2(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left, dtype=np.float32).reshape(-1) - np.asarray(right, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(diff) / max(np.sqrt(float(diff.size)), 1.0))


def _demo_row(demos: Any, demo_id: int) -> int:
    ids = np.asarray(demos.demo_ids, dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(ids == int(demo_id))
    if matches.size == 0:
        raise KeyError(f"Unknown demo id: {demo_id}")
    return int(matches[0])
