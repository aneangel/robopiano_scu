from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nocturne.repair_selected_path import repair_selected_path
from nocturne.schema import SegmentCandidate, StitchConfig, TransitionWeights


def _candidate(event_index: int, demo_id: int, score: float, avoidable_missed: int) -> SegmentCandidate:
    return SegmentCandidate(
        segment_id=event_index * 100 + demo_id,
        event_index=event_index,
        demo_id=demo_id,
        window_start=0,
        window_end=2,
        local_score=score,
        frame_f1=1.0,
        frame_precision=1.0,
        frame_recall=1.0,
        event_precision=1.0,
        event_recall=1.0,
        missed_keys=avoidable_missed,
        wrong_keys=0,
        timing_abs_error_frames=0.0,
        action_smoothness=0.0,
        joint_velocity=0.0,
        joint_acceleration=0.0,
        fingertip_jerk=0.0,
        target_keys=1,
        avoidable_missed_keys=avoidable_missed,
        clean_hit=avoidable_missed == 0,
    )


def test_repair_selected_path_swaps_to_remove_avoidable_error() -> None:
    demos = SimpleNamespace(
        demo_ids=np.asarray([0, 1], dtype=np.int64),
        num_frames=4,
        hand_joints=np.zeros((2, 4, 46), dtype=np.float32),
        hand_fingertips=np.zeros((2, 4, 30), dtype=np.float32),
        actions=np.zeros((2, 4, 39), dtype=np.float32),
    )
    selected = [_candidate(0, 0, 10.0, 1), _candidate(1, 0, 10.0, 0)]
    candidates = [[selected[0], _candidate(0, 1, 9.0, 0)], [selected[1]]]

    repaired, _, report = repair_selected_path(
        selected,
        candidates,
        demos,
        intervals=[(0, 2), (2, 4)],
        config=StitchConfig(objective_mode="correctness", repair_enabled=True, repair_transition_margin=10.0),
        weights=TransitionWeights(joint_position=0.0, fingertip_position=0.0, joint_velocity=0.0, action=0.0),
    )

    assert repaired[0].demo_id == 1
    assert report["num_swaps"] == 1


def test_repair_selected_path_skips_in_legacy_mode() -> None:
    demos = SimpleNamespace(
        demo_ids=np.asarray([0, 1], dtype=np.int64),
        num_frames=4,
        hand_joints=np.zeros((2, 4, 46), dtype=np.float32),
        hand_fingertips=np.zeros((2, 4, 30), dtype=np.float32),
        actions=np.zeros((2, 4, 39), dtype=np.float32),
    )
    selected = [_candidate(0, 0, 10.0, 1)]
    candidates = [[selected[0], _candidate(0, 1, 9.0, 0)]]

    repaired, _, report = repair_selected_path(
        selected,
        candidates,
        demos,
        intervals=[(0, 4)],
        config=StitchConfig(objective_mode="legacy", repair_enabled=True),
    )

    assert repaired[0].demo_id == 0
    assert report["status"] == "skipped"
