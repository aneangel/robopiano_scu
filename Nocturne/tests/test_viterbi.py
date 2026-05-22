from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nocturne.schema import SegmentCandidate, TransitionWeights
from nocturne.viterbi import viterbi_select


def _candidate(event_index: int, demo_id: int, score: float) -> SegmentCandidate:
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
        missed_keys=0,
        wrong_keys=0,
        timing_abs_error_frames=0.0,
        action_smoothness=0.0,
        joint_velocity=0.0,
        joint_acceleration=0.0,
        fingertip_jerk=0.0,
    )


def test_viterbi_trades_local_score_against_transition() -> None:
    demos = SimpleNamespace(
        demo_ids=np.asarray([0, 1], dtype=np.int64),
        num_frames=4,
        hand_joints=np.zeros((2, 4, 46), dtype=np.float32),
        hand_fingertips=np.zeros((2, 4, 30), dtype=np.float32),
        actions=np.zeros((2, 4, 39), dtype=np.float32),
    )
    demos.hand_joints[1, 2:] = 1.0
    demos.hand_fingertips[1, 2:] = 0.1
    candidates = [
        [_candidate(0, 0, 10.0), _candidate(0, 1, 9.9)],
        [_candidate(1, 0, 9.0), _candidate(1, 1, 10.0)],
    ]
    selected, costs, _ = viterbi_select(
        candidates,
        demos,
        intervals=[(0, 2), (2, 4)],
        weights=TransitionWeights(
            joint_position=10.0,
            fingertip_position=10.0,
            joint_velocity=0.0,
            action=0.0,
            hard_joint_jump=100.0,
            hard_fingertip_jump=100.0,
        ),
    )

    assert [item.demo_id for item in selected] == [0, 0]
    assert len(costs) == 1
