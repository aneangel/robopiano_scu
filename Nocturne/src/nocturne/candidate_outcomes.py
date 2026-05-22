from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nocturne.schema import NoteEvent, StitchConfig


@dataclass(frozen=True, slots=True)
class EventOutcomePrior:
    event_index: int
    target_keys: tuple[int, ...]
    num_demos: int
    hit_counts: dict[int, int]
    wrong_onset_counts: dict[int, int]
    interval_hit_counts: dict[int, int]
    interval_wrong_onset_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    hit_keys: tuple[int, ...]
    missed_keys: tuple[int, ...]
    wrong_keys: tuple[int, ...]
    avoidable_missed_keys: tuple[int, ...]
    unavoidable_missed_keys: tuple[int, ...]
    avoidable_wrong_keys: tuple[int, ...]
    all_demo_wrong_keys: tuple[int, ...]

    @property
    def clean_hit(self) -> bool:
        return not self.missed_keys and not self.wrong_keys


def build_event_outcome_priors(
    demos: Any,
    events: list[NoteEvent],
    config: StitchConfig,
    *,
    intervals: list[tuple[int, int]] | None = None,
) -> dict[int, EventOutcomePrior]:
    reference_goals = demos.goals[0]
    priors: dict[int, EventOutcomePrior] = {}
    for event in events:
        hit_counts = {int(key): 0 for key in event.keys}
        wrong_counts: dict[int, int] = {}
        interval_hit_counts = {int(key): 0 for key in event.keys}
        interval_wrong_counts: dict[int, int] = {}
        interval = None if intervals is None else intervals[int(event.event_index)]
        for row_index in range(int(demos.num_demos)):
            outcome = classify_candidate_outcome(
                event=event,
                reference_goals=reference_goals,
                demo_piano_states=demos.piano_states[row_index],
                config=config,
                prior=None,
            )
            for key in outcome.hit_keys:
                hit_counts[int(key)] = hit_counts.get(int(key), 0) + 1
            for key in set(int(key) for key in outcome.wrong_keys):
                wrong_counts[key] = wrong_counts.get(key, 0) + 1
            if interval is not None:
                interval_outcome = classify_interval_candidate_outcome(
                    event=event,
                    reference_goals=reference_goals,
                    demo_piano_states=demos.piano_states[row_index],
                    config=config,
                    interval=interval,
                    prior=None,
                )
                for key in set(int(key) for key in interval_outcome.hit_keys):
                    interval_hit_counts[key] = interval_hit_counts.get(key, 0) + 1
                for key in set(int(key) for key in interval_outcome.wrong_keys):
                    interval_wrong_counts[key] = interval_wrong_counts.get(key, 0) + 1
        priors[int(event.event_index)] = EventOutcomePrior(
            event_index=int(event.event_index),
            target_keys=tuple(int(key) for key in event.keys),
            num_demos=int(demos.num_demos),
            hit_counts=hit_counts,
            wrong_onset_counts=wrong_counts,
            interval_hit_counts=interval_hit_counts,
            interval_wrong_onset_counts=interval_wrong_counts,
        )
    return priors


def classify_candidate_outcome(
    *,
    event: NoteEvent,
    reference_goals: np.ndarray,
    demo_piano_states: np.ndarray,
    config: StitchConfig,
    prior: EventOutcomePrior | None,
) -> CandidateOutcome:
    active_played = np.asarray(demo_piano_states[:, :88], dtype=np.float32) > float(config.threshold)
    active_target = np.asarray(reference_goals[:, :88], dtype=np.float32) > float(config.threshold)
    onset = int(event.onset_frame)
    start = max(onset - int(config.event_tolerance_frames), 0)
    end = min(onset + int(config.event_tolerance_frames) + 1, active_played.shape[0])
    event_keys = tuple(int(key) for key in event.keys)
    target_context_keys = set(int(key) for key in np.flatnonzero(np.any(active_target[start:end], axis=0)).tolist())

    onset_frames = played_onset_frames_by_key(active_played, start, end)
    hits: list[int] = []
    missed: list[int] = []
    for key in event_keys:
        if int(key) in onset_frames:
            hits.append(int(key))
        else:
            missed.append(int(key))

    wrong = sorted(set(onset_frames) - set(event_keys) - target_context_keys)
    hit_counts = prior.hit_counts if prior is not None else {}
    wrong_counts = prior.wrong_onset_counts if prior is not None else {}
    num_demos = int(prior.num_demos) if prior is not None else 1
    avoidable_missed = tuple(int(key) for key in missed if int(hit_counts.get(int(key), 1)) > 0)
    unavoidable_missed = tuple(int(key) for key in missed if int(hit_counts.get(int(key), 1)) <= 0)
    avoidable_wrong = tuple(int(key) for key in wrong if int(wrong_counts.get(int(key), 0)) < num_demos)
    all_demo_wrong = tuple(int(key) for key in wrong if int(wrong_counts.get(int(key), 0)) >= num_demos)
    return CandidateOutcome(
        hit_keys=tuple(sorted(hits)),
        missed_keys=tuple(sorted(missed)),
        wrong_keys=tuple(sorted(int(key) for key in wrong)),
        avoidable_missed_keys=avoidable_missed,
        unavoidable_missed_keys=unavoidable_missed,
        avoidable_wrong_keys=avoidable_wrong,
        all_demo_wrong_keys=all_demo_wrong,
    )


def classify_interval_candidate_outcome(
    *,
    event: NoteEvent,
    reference_goals: np.ndarray,
    demo_piano_states: np.ndarray,
    config: StitchConfig,
    interval: tuple[int, int],
    prior: EventOutcomePrior | None,
) -> CandidateOutcome:
    active_played = np.asarray(demo_piano_states[:, :88], dtype=np.float32) > float(config.threshold)
    active_target = np.asarray(reference_goals[:, :88], dtype=np.float32) > float(config.threshold)
    start, end = int(interval[0]), int(interval[1])
    target_onsets = _target_onsets(active_target, start, end)
    played_onsets = _onset_events(active_played, start, end)

    used_played: set[int] = set()
    hit_keys: list[int] = []
    missed_keys: list[int] = []
    tolerance = int(config.event_tolerance_frames)
    for target_frame, target_key in target_onsets:
        best_index = None
        best_abs = 10**9
        for played_index, (played_frame, played_key) in enumerate(played_onsets):
            if played_index in used_played or int(played_key) != int(target_key):
                continue
            err = abs(int(played_frame) - int(target_frame))
            if err <= tolerance and err < best_abs:
                best_abs = err
                best_index = played_index
        if best_index is None:
            missed_keys.append(int(target_key))
        else:
            used_played.add(best_index)
            hit_keys.append(int(target_key))

    wrong_keys = [int(key) for index, (_, key) in enumerate(played_onsets) if index not in used_played]
    hit_counts = prior.interval_hit_counts if prior is not None else {}
    wrong_counts = prior.interval_wrong_onset_counts if prior is not None else {}
    num_demos = int(prior.num_demos) if prior is not None else 1
    avoidable_missed = tuple(int(key) for key in missed_keys if int(hit_counts.get(int(key), 1)) > 0)
    unavoidable_missed = tuple(int(key) for key in missed_keys if int(hit_counts.get(int(key), 1)) <= 0)
    avoidable_wrong = tuple(int(key) for key in wrong_keys if int(wrong_counts.get(int(key), 0)) < num_demos)
    all_demo_wrong = tuple(int(key) for key in wrong_keys if int(wrong_counts.get(int(key), 0)) >= num_demos)
    return CandidateOutcome(
        hit_keys=tuple(sorted(hit_keys)),
        missed_keys=tuple(sorted(missed_keys)),
        wrong_keys=tuple(sorted(wrong_keys)),
        avoidable_missed_keys=avoidable_missed,
        unavoidable_missed_keys=unavoidable_missed,
        avoidable_wrong_keys=avoidable_wrong,
        all_demo_wrong_keys=all_demo_wrong,
    )


def correctness_error_count(candidate: Any) -> int:
    return int(candidate.avoidable_missed_keys) + int(candidate.avoidable_wrong_keys)


def correctness_severity(candidate: Any) -> tuple[int, int, int, int]:
    return (
        int(candidate.avoidable_missed_keys),
        int(candidate.avoidable_wrong_keys),
        int(candidate.unavoidable_missed_keys),
        int(candidate.all_demo_wrong_keys),
    )


def strict_correctness_severity(candidate: Any) -> tuple[int, int, int, int, int, int]:
    avoidable_missed = int(getattr(candidate, "avoidable_interval_missed_keys", 0))
    avoidable_wrong = int(getattr(candidate, "avoidable_interval_wrong_keys", 0))
    unavoidable_missed = int(getattr(candidate, "unavoidable_interval_missed_keys", 0))
    all_demo_wrong = int(getattr(candidate, "all_demo_interval_wrong_keys", 0))
    return (
        avoidable_missed + avoidable_wrong,
        avoidable_missed,
        avoidable_wrong,
        unavoidable_missed + all_demo_wrong,
        unavoidable_missed,
        all_demo_wrong,
    )


def played_onset_frames_by_key(active_played: np.ndarray, start: int, end: int) -> dict[int, list[int]]:
    frames_by_key: dict[int, list[int]] = {}
    total = int(active_played.shape[0])
    left = max(int(start), 0)
    right = min(int(end), total)
    for frame in range(left, right):
        previous = active_played[frame - 1] if frame > 0 else np.zeros((active_played.shape[1],), dtype=bool)
        onsets = np.flatnonzero(active_played[frame] & ~previous)
        for key in onsets.tolist():
            frames_by_key.setdefault(int(key), []).append(int(frame))
    return frames_by_key


def _target_onsets(active_target: np.ndarray, start: int, end: int) -> list[tuple[int, int]]:
    return _onset_events(active_target, start, end)


def _onset_events(active: np.ndarray, start: int, end: int) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    total = int(active.shape[0])
    left = max(int(start), 0)
    right = min(int(end), total)
    for frame in range(left, right):
        previous = active[frame - 1] if frame > 0 else np.zeros((active.shape[1],), dtype=bool)
        onsets = np.flatnonzero(active[frame] & ~previous)
        events.extend((int(frame), int(key)) for key in onsets.tolist())
    return events
