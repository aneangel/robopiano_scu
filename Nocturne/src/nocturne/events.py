from __future__ import annotations

import numpy as np

from nocturne.schema import NoteEvent


def piano_roll_88(roll: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    array = np.asarray(roll, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected roll [T, K], got {array.shape}")
    return array[:, :88] > float(threshold)


def extract_note_events(
    goals: np.ndarray,
    *,
    threshold: float = 0.5,
    chord_tolerance_frames: int = 1,
) -> list[NoteEvent]:
    active = piano_roll_88(goals, threshold=threshold)
    previous = np.zeros((active.shape[1],), dtype=bool)
    onsets: list[tuple[int, int, int]] = []
    for frame, row in enumerate(active):
        keys = np.flatnonzero(row & ~previous)
        for key in keys.tolist():
            end = _release_frame(active[:, int(key)], frame)
            onsets.append((int(frame), int(end), int(key)))
        previous = row
    if not onsets:
        return []

    grouped: list[list[tuple[int, int, int]]] = []
    tolerance = max(int(chord_tolerance_frames), 0)
    for item in sorted(onsets, key=lambda value: (value[0], value[2])):
        if not grouped or item[0] - grouped[-1][-1][0] > tolerance:
            grouped.append([item])
        else:
            grouped[-1].append(item)

    events: list[NoteEvent] = []
    for index, group in enumerate(grouped):
        onset = min(item[0] for item in group)
        end = max(item[1] for item in group)
        keys = tuple(sorted({item[2] for item in group}))
        events.append(NoteEvent(event_index=index, onset_frame=int(onset), end_frame=int(end), keys=keys))
    return events


def event_intervals(events: list[NoteEvent], num_frames: int) -> list[tuple[int, int]]:
    if not events:
        return []
    onsets = np.asarray([event.onset_frame for event in events], dtype=np.int64)
    intervals: list[tuple[int, int]] = []
    for index, event in enumerate(events):
        if index == 0:
            start = 0
        else:
            start = int((onsets[index - 1] + onsets[index]) // 2)
        if index == len(events) - 1:
            end = int(num_frames)
        else:
            end = int((onsets[index] + onsets[index + 1]) // 2)
        intervals.append((max(start, 0), min(max(end, start + 1), int(num_frames))))
    return intervals


def press_frame_indices(events: list[NoteEvent]) -> np.ndarray:
    return np.asarray(sorted({int(event.onset_frame) for event in events}), dtype=np.int64)


def event_key_mask(events: list[NoteEvent], *, num_keys: int = 88) -> np.ndarray:
    mask = np.zeros((len(events), int(num_keys)), dtype=np.float32)
    for event in events:
        for key in event.keys:
            if 0 <= int(key) < int(num_keys):
                mask[int(event.event_index), int(key)] = 1.0
    return mask


def protected_press_mask(
    goals: np.ndarray,
    press_frames: np.ndarray,
    *,
    radius: int,
    threshold: float = 0.5,
) -> np.ndarray:
    active = np.any(piano_roll_88(goals, threshold=threshold), axis=1)
    protected = active.copy()
    total = int(protected.shape[0])
    for frame in np.asarray(press_frames, dtype=np.int64).reshape(-1):
        start = max(int(frame) - int(radius), 0)
        end = min(int(frame) + int(radius) + 1, total)
        protected[start:end] = True
    return protected


def goals_are_compatible(reference: np.ndarray, candidate: np.ndarray, *, threshold: float = 0.5) -> bool:
    return bool(np.array_equal(piano_roll_88(reference, threshold=threshold), piano_roll_88(candidate, threshold=threshold)))


def _release_frame(active_key: np.ndarray, onset_frame: int) -> int:
    frame = int(onset_frame) + 1
    while frame < active_key.shape[0] and bool(active_key[frame]):
        frame += 1
    return max(frame, int(onset_frame) + 1)
