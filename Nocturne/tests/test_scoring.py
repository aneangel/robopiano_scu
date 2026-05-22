from __future__ import annotations

import numpy as np

from nocturne.events import extract_note_events
from nocturne.schema import StitchConfig
from nocturne.scoring import filter_candidates_for_selection, score_candidate


def _candidate_score(played: np.ndarray) -> float:
    goals = np.zeros((16, 89), dtype=np.float32)
    goals[5:8, 10] = 1.0
    event = extract_note_events(goals)[0]
    zeros_action = np.zeros((16, 39), dtype=np.float32)
    zeros_joints = np.zeros((16, 46), dtype=np.float32)
    zeros_tips = np.zeros((16, 30), dtype=np.float32)
    candidate = score_candidate(
        event=event,
        demo_id=0,
        reference_goals=goals,
        demo_piano_states=played,
        demo_actions=zeros_action,
        demo_joints=zeros_joints,
        demo_fingertips=zeros_tips,
        config=StitchConfig(window_frames=8, pre_frames=3, event_tolerance_frames=1),
    )
    return candidate.local_score


def test_scoring_ranks_perfect_above_missed_and_wrong() -> None:
    perfect = np.zeros((16, 89), dtype=np.float32)
    perfect[5:8, 10] = 1.0
    missed = np.zeros((16, 89), dtype=np.float32)
    wrong = perfect.copy()
    wrong[5:8, 20] = 1.0

    assert _candidate_score(perfect) > _candidate_score(wrong) > _candidate_score(missed)


def test_strict_filter_keeps_only_minimum_interval_error_candidates() -> None:
    goals = np.zeros((16, 89), dtype=np.float32)
    goals[5:8, 10] = 1.0
    event = extract_note_events(goals)[0]
    zeros_action = np.zeros((16, 39), dtype=np.float32)
    zeros_joints = np.zeros((16, 46), dtype=np.float32)
    zeros_tips = np.zeros((16, 30), dtype=np.float32)
    clean = np.zeros((16, 89), dtype=np.float32)
    clean[5:8, 10] = 1.0
    wrong = clean.copy()
    wrong[10, 20] = 1.0
    config = StitchConfig(objective_mode="strict", window_frames=8, pre_frames=3, event_tolerance_frames=1)
    clean_candidate = score_candidate(
        event=event,
        demo_id=0,
        reference_goals=goals,
        demo_piano_states=clean,
        demo_actions=zeros_action,
        demo_joints=zeros_joints,
        demo_fingertips=zeros_tips,
        config=config,
        interval=(3, 12),
    )
    wrong_candidate = score_candidate(
        event=event,
        demo_id=1,
        reference_goals=goals,
        demo_piano_states=wrong,
        demo_actions=zeros_action,
        demo_joints=zeros_joints,
        demo_fingertips=zeros_tips,
        config=config,
        interval=(3, 12),
    )

    filtered, report = filter_candidates_for_selection([[clean_candidate, wrong_candidate]], config)

    assert filtered == [[clean_candidate]]
    assert report["num_removed"] == 1
