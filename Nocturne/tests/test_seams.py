from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nocturne.schema import NoteEvent, SegmentCandidate, StitchConfig
from nocturne.seams import adaptive_min_distance_intervals, smooth_stitched_payload


def _candidate(event_index: int, demo_id: int) -> SegmentCandidate:
    return SegmentCandidate(
        segment_id=event_index * 100 + demo_id,
        event_index=event_index,
        demo_id=demo_id,
        window_start=0,
        window_end=10,
        local_score=0.0,
        frame_f1=1.0,
        frame_precision=1.0,
        frame_recall=1.0,
        event_precision=1.0,
        event_recall=1.0,
        missed_keys=0,
        wrong_keys=0,
        timing_abs_error_frames=0.0,
        action_smoothness=0.0,
        joint_velocity=0.0,
        joint_acceleration=0.0,
        fingertip_jerk=0.0,
    )


def test_smoothing_preserves_protected_press_frames() -> None:
    goals = np.zeros((10, 89), dtype=np.float32)
    goals[5, 10] = 1.0
    joints = np.zeros((10, 46), dtype=np.float32)
    joints[5] = 99.0
    joints[6:] = 10.0
    payload = {
        "goals": goals,
        "hand_joints": joints.copy(),
        "hand_fingertips": np.zeros((10, 30), dtype=np.float32),
        "actions": np.zeros((10, 39), dtype=np.float32),
    }

    smoothed = smooth_stitched_payload(payload, seam_frames=[6], press_frames=np.asarray([5]), blend_radius=3)

    assert np.allclose(smoothed["hand_joints"][5], joints[5])


def test_adaptive_min_distance_intervals_moves_seam_to_closest_hand_state() -> None:
    demos = SimpleNamespace(
        demo_ids=np.asarray([0, 1], dtype=np.int64),
        num_frames=10,
        hand_joints=np.zeros((2, 10, 2), dtype=np.float32),
        hand_fingertips=np.zeros((2, 10, 2), dtype=np.float32),
        actions=np.zeros((2, 10, 2), dtype=np.float32),
    )
    demos.hand_joints[1] = 10.0
    demos.hand_joints[0, 6] = 3.0
    demos.hand_joints[1, 7] = 3.0
    selected = [_candidate(0, 0), _candidate(1, 1)]
    events = [
        NoteEvent(event_index=0, onset_frame=2, end_frame=3, keys=(10,)),
        NoteEvent(event_index=1, onset_frame=8, end_frame=9, keys=(11,)),
    ]

    intervals, report = adaptive_min_distance_intervals(
        demos,
        selected,
        events,
        intervals=[(0, 5), (5, 10)],
        config=StitchConfig(adaptive_seam_enabled=True, seam_search_margin_frames=1),
    )

    assert intervals == [(0, 7), (7, 10)]
    assert report["num_changed"] == 1


def test_adaptive_min_distance_intervals_can_be_disabled() -> None:
    demos = SimpleNamespace(
        demo_ids=np.asarray([0, 1], dtype=np.int64),
        num_frames=10,
        hand_joints=np.zeros((2, 10, 2), dtype=np.float32),
        hand_fingertips=np.zeros((2, 10, 2), dtype=np.float32),
        actions=np.zeros((2, 10, 2), dtype=np.float32),
    )
    selected = [_candidate(0, 0), _candidate(1, 1)]
    events = [
        NoteEvent(event_index=0, onset_frame=2, end_frame=3, keys=(10,)),
        NoteEvent(event_index=1, onset_frame=8, end_frame=9, keys=(11,)),
    ]

    intervals, report = adaptive_min_distance_intervals(
        demos,
        selected,
        events,
        intervals=[(0, 5), (5, 10)],
        config=StitchConfig(adaptive_seam_enabled=False),
    )

    assert intervals == [(0, 5), (5, 10)]
    assert report["status"] == "disabled"
