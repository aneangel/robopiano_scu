from __future__ import annotations

import numpy as np

from nocturne.candidate_outcomes import (
    EventOutcomePrior,
    build_event_outcome_priors,
    classify_candidate_outcome,
    classify_interval_candidate_outcome,
    played_onset_frames_by_key,
    strict_correctness_severity,
)
from nocturne.offline_eval import frame_metrics
from nocturne.schema import NoteEvent, SegmentCandidate, StitchConfig


def candidate_window(event: NoteEvent, num_frames: int, config: StitchConfig) -> tuple[int, int]:
    start = max(int(event.onset_frame) - int(config.pre_frames), 0)
    end = min(start + int(config.window_frames), int(num_frames))
    if end - start < int(config.window_frames):
        start = max(end - int(config.window_frames), 0)
    return int(start), int(end)


def score_candidate(
    *,
    event: NoteEvent,
    demo_id: int,
    reference_goals: np.ndarray,
    demo_piano_states: np.ndarray,
    demo_actions: np.ndarray,
    demo_joints: np.ndarray,
    demo_fingertips: np.ndarray,
    config: StitchConfig,
    event_prior: EventOutcomePrior | None = None,
    interval: tuple[int, int] | None = None,
) -> SegmentCandidate:
    start, end = candidate_window(event, reference_goals.shape[0], config)
    target = np.asarray(reference_goals[start:end, :88], dtype=np.float32)
    played = np.asarray(demo_piano_states[start:end, :88], dtype=np.float32)
    fm = frame_metrics(target, played, threshold=config.threshold)
    event_stats = _event_hit_stats(
        event=event,
        played=demo_piano_states,
        reference_goals=reference_goals,
        tolerance=int(config.event_tolerance_frames),
        threshold=float(config.threshold),
    )
    smooth = _motion_stats(
        actions=demo_actions[start:end],
        joints=demo_joints[start:end],
        fingertips=demo_fingertips[start:end],
        dt=float(config.dt),
    )
    legacy_score = (
        550.0 * fm["frame_f1"]
        + 220.0 * event_stats["event_recall"]
        + 120.0 * event_stats["event_precision"]
        - 220.0 * event_stats["missed_keys"]
        - 45.0 * event_stats["wrong_keys"]
        - 8.0 * event_stats["timing_abs_error_frames"]
        - 0.05 * smooth["action_smoothness"]
        - 0.01 * smooth["joint_velocity"]
        - 0.002 * smooth["joint_acceleration"]
        - 0.01 * smooth["fingertip_jerk"]
    )
    outcome = classify_candidate_outcome(
        event=event,
        reference_goals=reference_goals,
        demo_piano_states=demo_piano_states,
        config=config,
        prior=event_prior,
    )
    interval_outcome = (
        classify_interval_candidate_outcome(
            event=event,
            reference_goals=reference_goals,
            demo_piano_states=demo_piano_states,
            config=config,
            interval=interval,
            prior=event_prior,
        )
        if interval is not None
        else outcome
    )
    if str(config.objective_mode) == "legacy":
        local_score = legacy_score
    else:
        correctness_penalty = (
            float(config.avoidable_miss_cost) * len(outcome.avoidable_missed_keys)
            + float(config.avoidable_wrong_cost) * len(outcome.avoidable_wrong_keys)
            + float(config.unavoidable_miss_cost) * len(outcome.unavoidable_missed_keys)
            + float(config.all_demo_wrong_cost) * len(outcome.all_demo_wrong_keys)
            + float(config.avoidable_miss_cost) * len(interval_outcome.avoidable_missed_keys)
            + float(config.avoidable_wrong_cost) * len(interval_outcome.avoidable_wrong_keys)
            + float(config.unavoidable_miss_cost) * len(interval_outcome.unavoidable_missed_keys)
            + float(config.all_demo_wrong_cost) * len(interval_outcome.all_demo_wrong_keys)
        )
        correctness_bonus = (
            1800.0 * len(outcome.hit_keys)
            + 800.0 * len(interval_outcome.hit_keys)
            + (2500.0 if outcome.clean_hit else 0.0)
            + (1200.0 if interval_outcome.clean_hit else 0.0)
        )
        local_score = legacy_score + correctness_bonus - correctness_penalty
    segment_id = int(event.event_index) * 100000 + int(demo_id)
    return SegmentCandidate(
        segment_id=segment_id,
        event_index=int(event.event_index),
        demo_id=int(demo_id),
        window_start=start,
        window_end=end,
        local_score=float(local_score),
        frame_f1=float(fm["frame_f1"]),
        frame_precision=float(fm["frame_precision"]),
        frame_recall=float(fm["frame_recall"]),
        event_precision=float(event_stats["event_precision"]),
        event_recall=float(event_stats["event_recall"]),
        missed_keys=int(event_stats["missed_keys"]),
        wrong_keys=int(event_stats["wrong_keys"]),
        timing_abs_error_frames=float(event_stats["timing_abs_error_frames"]),
        action_smoothness=float(smooth["action_smoothness"]),
        joint_velocity=float(smooth["joint_velocity"]),
        joint_acceleration=float(smooth["joint_acceleration"]),
        fingertip_jerk=float(smooth["fingertip_jerk"]),
        hit_keys=int(len(outcome.hit_keys)),
        target_keys=int(len(event.keys)),
        avoidable_missed_keys=int(len(outcome.avoidable_missed_keys)),
        unavoidable_missed_keys=int(len(outcome.unavoidable_missed_keys)),
        avoidable_wrong_keys=int(len(outcome.avoidable_wrong_keys)),
        all_demo_wrong_keys=int(len(outcome.all_demo_wrong_keys)),
        interval_missed_keys=int(len(interval_outcome.missed_keys)),
        interval_wrong_keys=int(len(interval_outcome.wrong_keys)),
        avoidable_interval_missed_keys=int(len(interval_outcome.avoidable_missed_keys)),
        unavoidable_interval_missed_keys=int(len(interval_outcome.unavoidable_missed_keys)),
        avoidable_interval_wrong_keys=int(len(interval_outcome.avoidable_wrong_keys)),
        all_demo_interval_wrong_keys=int(len(interval_outcome.all_demo_wrong_keys)),
        clean_hit=bool(outcome.clean_hit),
        interval_clean=bool(interval_outcome.clean_hit),
    )


def build_candidates(
    demos: object,
    events: list[NoteEvent],
    config: StitchConfig,
    *,
    intervals: list[tuple[int, int]] | None = None,
) -> list[list[SegmentCandidate]]:
    reference_goals = demos.goals[0]
    priors = (
        build_event_outcome_priors(demos, events, config, intervals=intervals)
        if str(config.objective_mode) != "legacy"
        else {}
    )
    by_event: list[list[SegmentCandidate]] = []
    for event in events:
        event_candidates = []
        interval = None if intervals is None else intervals[int(event.event_index)]
        for row_index, demo_id in enumerate(np.asarray(demos.demo_ids).reshape(-1).tolist()):
            event_candidates.append(
                score_candidate(
                    event=event,
                    demo_id=int(demo_id),
                    reference_goals=reference_goals,
                    demo_piano_states=demos.piano_states[row_index],
                    demo_actions=demos.actions[row_index],
                    demo_joints=demos.hand_joints[row_index],
                    demo_fingertips=demos.hand_fingertips[row_index],
                    config=config,
                    event_prior=priors.get(int(event.event_index)),
                    interval=interval,
                )
            )
        by_event.append(event_candidates)
    return by_event


def flatten_candidates(candidates_by_event: list[list[SegmentCandidate]]) -> list[SegmentCandidate]:
    return [candidate for group in candidates_by_event for candidate in group]


def filter_candidates_for_selection(
    candidates_by_event: list[list[SegmentCandidate]],
    config: StitchConfig,
) -> tuple[list[list[SegmentCandidate]], dict[str, object]]:
    if str(config.objective_mode) != "strict":
        return candidates_by_event, {"mode": str(config.objective_mode), "num_removed": 0}
    filtered: list[list[SegmentCandidate]] = []
    removed = 0
    event_rows: list[dict[str, object]] = []
    for event_index, group in enumerate(candidates_by_event):
        if not group:
            filtered.append(group)
            continue
        best = min(strict_correctness_severity(candidate) for candidate in group)
        kept = [candidate for candidate in group if strict_correctness_severity(candidate) == best]
        filtered.append(kept)
        removed += len(group) - len(kept)
        event_rows.append(
            {
                "event_index": int(event_index),
                "num_candidates": int(len(group)),
                "num_kept": int(len(kept)),
                "best_strict_severity": list(best),
            }
        )
    return filtered, {
        "mode": "strict",
        "num_removed": int(removed),
        "num_events_filtered": int(sum(1 for row in event_rows if int(row["num_kept"]) < int(row["num_candidates"]))),
        "events": event_rows[:200],
    }


def _event_hit_stats(
    *,
    event: NoteEvent,
    played: np.ndarray,
    reference_goals: np.ndarray,
    tolerance: int,
    threshold: float,
) -> dict[str, float]:
    active_played = np.asarray(played[:, :88], dtype=np.float32) > float(threshold)
    active_target = np.asarray(reference_goals[:, :88], dtype=np.float32) > float(threshold)
    onset = int(event.onset_frame)
    start = max(onset - int(tolerance), 0)
    end = min(onset + int(tolerance) + 1, active_played.shape[0])
    onset_frames = played_onset_frames_by_key(active_played, start, end)
    hits = 0
    errors = []
    for key in event.keys:
        frames = onset_frames.get(int(key), [])
        if frames:
            hits += 1
            errors.append(float(min(abs(int(frame) - onset) for frame in frames)))
        else:
            errors.append(float(tolerance + 1))
    event_keys = set(int(key) for key in event.keys)
    played_keys = set(int(key) for key in onset_frames)
    target_context_keys = set(int(key) for key in np.flatnonzero(np.any(active_target[start:end], axis=0)).tolist())
    wrong_keys = len(played_keys - event_keys - target_context_keys)
    missed = max(len(event.keys) - hits, 0)
    precision = float(hits / max(hits + wrong_keys, 1))
    recall = float(hits / max(len(event.keys), 1))
    return {
        "event_precision": precision,
        "event_recall": recall,
        "missed_keys": float(missed),
        "wrong_keys": float(wrong_keys),
        "timing_abs_error_frames": float(np.mean(errors)) if errors else 0.0,
    }


def _motion_stats(actions: np.ndarray, joints: np.ndarray, fingertips: np.ndarray, *, dt: float) -> dict[str, float]:
    dt = max(float(dt), 1e-6)
    action_diff = np.diff(np.asarray(actions, dtype=np.float32), axis=0)
    joint_vel = np.diff(np.asarray(joints, dtype=np.float32), axis=0) / dt
    joint_acc = np.diff(joint_vel, axis=0) / dt if joint_vel.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    tips = np.asarray(fingertips, dtype=np.float32)
    tip_vel = np.diff(tips, axis=0) / dt if tips.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    tip_acc = np.diff(tip_vel, axis=0) / dt if tip_vel.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    tip_jerk = np.diff(tip_acc, axis=0) / dt if tip_acc.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    return {
        "action_smoothness": float(np.mean(action_diff * action_diff)) if action_diff.size else 0.0,
        "joint_velocity": _mean_norm(joint_vel),
        "joint_acceleration": _mean_norm(joint_acc),
        "fingertip_jerk": _mean_norm(tip_jerk),
    }


def _mean_norm(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)))
