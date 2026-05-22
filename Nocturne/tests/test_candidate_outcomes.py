from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nocturne.candidate_outcomes import (
    build_event_outcome_priors,
    classify_candidate_outcome,
    classify_interval_candidate_outcome,
)
from nocturne.events import extract_note_events
from nocturne.schema import StitchConfig


@dataclass
class FakeDemos:
    demo_ids: np.ndarray
    goals: np.ndarray
    piano_states: np.ndarray

    @property
    def num_demos(self) -> int:
        return int(self.goals.shape[0])


def test_candidate_outcome_distinguishes_avoidable_and_unavoidable_errors() -> None:
    goals = np.zeros((2, 10, 89), dtype=np.float32)
    goals[:, 4:6, 10] = 1.0
    piano = np.zeros_like(goals)
    piano[0, 4:6, 10] = 1.0
    piano[0, 4:6, 20] = 1.0
    piano[1, 4:6, 20] = 1.0
    demos = FakeDemos(demo_ids=np.asarray([0, 1]), goals=goals, piano_states=piano)
    event = extract_note_events(goals[0])[0]
    config = StitchConfig(event_tolerance_frames=1)
    prior = build_event_outcome_priors(demos, [event], config)[0]

    missed_outcome = classify_candidate_outcome(
        event=event,
        reference_goals=goals[0],
        demo_piano_states=piano[1],
        config=config,
        prior=prior,
    )

    assert missed_outcome.avoidable_missed_keys == (10,)
    assert missed_outcome.all_demo_wrong_keys == (20,)
    assert not missed_outcome.clean_hit


def test_candidate_outcome_requires_press_onset_not_preheld_key() -> None:
    goals = np.zeros((1, 10, 89), dtype=np.float32)
    goals[:, 4:6, 10] = 1.0
    piano = np.zeros_like(goals)
    piano[:, 1:6, 10] = 1.0
    demos = FakeDemos(demo_ids=np.asarray([0]), goals=goals, piano_states=piano)
    event = extract_note_events(goals[0])[0]
    config = StitchConfig(event_tolerance_frames=1)
    prior = build_event_outcome_priors(demos, [event], config)[0]

    outcome = classify_candidate_outcome(
        event=event,
        reference_goals=goals[0],
        demo_piano_states=piano[0],
        config=config,
        prior=prior,
    )

    assert outcome.missed_keys == (10,)
    assert not outcome.clean_hit


def test_interval_candidate_outcome_counts_wrong_onset_outside_event_window() -> None:
    goals = np.zeros((1, 12, 89), dtype=np.float32)
    goals[:, 5:7, 10] = 1.0
    piano = np.zeros_like(goals)
    piano[:, 5:7, 10] = 1.0
    piano[:, 9:10, 20] = 1.0
    demos = FakeDemos(demo_ids=np.asarray([0]), goals=goals, piano_states=piano)
    event = extract_note_events(goals[0])[0]
    config = StitchConfig(event_tolerance_frames=1)
    prior = build_event_outcome_priors(demos, [event], config, intervals=[(3, 11)])[0]

    outcome = classify_interval_candidate_outcome(
        event=event,
        reference_goals=goals[0],
        demo_piano_states=piano[0],
        config=config,
        interval=(3, 11),
        prior=prior,
    )

    assert outcome.hit_keys == (10,)
    assert outcome.wrong_keys == (20,)
    assert outcome.all_demo_wrong_keys == (20,)
