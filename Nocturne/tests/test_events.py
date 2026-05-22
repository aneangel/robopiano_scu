from __future__ import annotations

import numpy as np

from nocturne.events import event_intervals, extract_note_events


def test_extract_note_events_groups_chords() -> None:
    goals = np.zeros((12, 89), dtype=np.float32)
    goals[2:5, 10] = 1.0
    goals[3:5, 14] = 1.0
    goals[8:10, 30] = 1.0

    events = extract_note_events(goals, chord_tolerance_frames=1)

    assert len(events) == 2
    assert events[0].onset_frame == 2
    assert events[0].keys == (10, 14)
    assert events[1].onset_frame == 8
    assert events[1].keys == (30,)


def test_event_intervals_cover_frames_once() -> None:
    goals = np.zeros((20, 89), dtype=np.float32)
    goals[2, 1] = 1.0
    goals[10, 2] = 1.0
    goals[18, 3] = 1.0
    events = extract_note_events(goals)

    intervals = event_intervals(events, 20)

    assert intervals[0][0] == 0
    assert intervals[-1][1] == 20
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    assert sum(end - start for start, end in intervals) == 20
