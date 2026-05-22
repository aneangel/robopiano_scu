from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nocturne.error_analysis import analyze_demo_error_consensus, analyze_stitched_error_context


@dataclass
class FakeDemos:
    demo_ids: np.ndarray
    goals: np.ndarray
    piano_states: np.ndarray

    @property
    def num_demos(self) -> int:
        return int(self.goals.shape[0])


def test_demo_error_consensus_marks_all_demo_misses_and_common_mispresses() -> None:
    goals = np.zeros((3, 8, 89), dtype=np.float32)
    goals[:, 2:4, 10] = 1.0
    piano = np.zeros_like(goals)
    piano[:, 2:4, 11] = 1.0
    demos = FakeDemos(demo_ids=np.asarray([0, 1, 2]), goals=goals, piano_states=piano)

    out = analyze_demo_error_consensus(demos, song_name="song", dt=0.05, tolerance_s=0.05)

    assert out["missed_target_events"][0]["target_key"] == 10
    assert out["missed_target_events"][0]["missed_by_all_demos"]
    assert out["mispress_buckets"][0]["played_key"] == 11
    assert out["mispress_buckets"][0]["seen_in_all_demos"]


def test_stitched_error_context_marks_avoidable_miss() -> None:
    goals = np.zeros((2, 8, 89), dtype=np.float32)
    goals[:, 2:4, 10] = 1.0
    piano = np.zeros_like(goals)
    piano[0, 2:4, 10] = 1.0
    demos = FakeDemos(demo_ids=np.asarray([0, 1]), goals=goals, piano_states=piano)
    stitched_piano = np.zeros((8, 89), dtype=np.float32)

    out = analyze_stitched_error_context(
        demos,
        song_name="song",
        stitched_goals=goals[0],
        stitched_piano_states=stitched_piano,
        dt=0.05,
        tolerance_s=0.05,
    )

    assert out["stitched_missed_events"][0]["target_key"] == 10
    assert out["stitched_missed_events"][0]["avoidable_from_raw_demos"]
    assert not out["stitched_missed_events"][0]["missed_by_all_raw_demos"]
