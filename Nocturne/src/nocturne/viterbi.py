from __future__ import annotations

import math
from typing import Any

import numpy as np

from nocturne.schema import SegmentCandidate, TransitionWeights


def transition_cost(
    previous: SegmentCandidate,
    current: SegmentCandidate,
    demos: Any,
    intervals: list[tuple[int, int]],
    *,
    weights: TransitionWeights | None = None,
    contact_mask: np.ndarray | None = None,
) -> float:
    weights = weights or TransitionWeights()
    seam = int(intervals[int(previous.event_index)][1])
    if seam <= 0 or seam >= int(demos.num_frames):
        return 0.0
    prev_row = _demo_row(demos, previous.demo_id)
    curr_row = _demo_row(demos, current.demo_id)
    left = seam - 1
    right = seam
    joint_jump = _l2(demos.hand_joints[prev_row, left] - demos.hand_joints[curr_row, right])
    fingertip_jump = _l2(demos.hand_fingertips[prev_row, left] - demos.hand_fingertips[curr_row, right])
    if joint_jump > float(weights.hard_joint_jump) or fingertip_jump > float(weights.hard_fingertip_jump):
        return math.inf
    prev_vel = _velocity_at(demos.hand_joints[prev_row], left)
    curr_vel = _velocity_at(demos.hand_joints[curr_row], right)
    velocity_jump = _l2(prev_vel - curr_vel)
    action_jump = _l2(demos.actions[prev_row, left] - demos.actions[curr_row, right])
    cost = (
        float(weights.joint_position) * joint_jump
        + float(weights.joint_velocity) * velocity_jump
        + float(weights.fingertip_position) * fingertip_jump
        + float(weights.action) * action_jump
    )
    if int(previous.demo_id) != int(current.demo_id):
        cost += float(weights.source_switch)
    if contact_mask is not None and bool(np.asarray(contact_mask, dtype=bool)[max(min(seam, len(contact_mask) - 1), 0)]):
        cost *= float(weights.contact_multiplier)
    return float(cost)


def viterbi_select(
    candidates_by_event: list[list[SegmentCandidate]],
    demos: Any,
    intervals: list[tuple[int, int]],
    *,
    weights: TransitionWeights | None = None,
    contact_mask: np.ndarray | None = None,
) -> tuple[list[SegmentCandidate], list[float], dict[str, Any]]:
    if not candidates_by_event:
        return [], [], {"status": "empty"}
    weights = weights or TransitionWeights()
    scores: list[np.ndarray] = []
    backptrs: list[np.ndarray] = []
    first = np.asarray([candidate.local_score for candidate in candidates_by_event[0]], dtype=np.float64)
    scores.append(first)
    backptrs.append(np.full((len(candidates_by_event[0]),), -1, dtype=np.int64))

    transition_layers: list[np.ndarray] = []
    for event_index in range(1, len(candidates_by_event)):
        prev_group = candidates_by_event[event_index - 1]
        curr_group = candidates_by_event[event_index]
        layer = np.full((len(prev_group), len(curr_group)), np.inf, dtype=np.float64)
        current_scores = np.full((len(curr_group),), -np.inf, dtype=np.float64)
        current_back = np.full((len(curr_group),), -1, dtype=np.int64)
        for curr_i, curr in enumerate(curr_group):
            best_score = -np.inf
            best_prev = -1
            for prev_i, prev in enumerate(prev_group):
                cost = transition_cost(prev, curr, demos, intervals, weights=weights, contact_mask=contact_mask)
                layer[prev_i, curr_i] = cost
                if not np.isfinite(cost):
                    continue
                value = float(scores[-1][prev_i]) + float(curr.local_score) - float(cost)
                if value > best_score:
                    best_score = value
                    best_prev = prev_i
            if best_prev < 0:
                local_best_prev = int(np.argmax(scores[-1]))
                best_prev = local_best_prev
                best_score = float(scores[-1][local_best_prev]) + float(curr.local_score)
            current_scores[curr_i] = best_score
            current_back[curr_i] = best_prev
        transition_layers.append(layer)
        scores.append(current_scores)
        backptrs.append(current_back)

    selected_indices = [int(np.argmax(scores[-1]))]
    for event_index in range(len(candidates_by_event) - 1, 0, -1):
        selected_indices.append(int(backptrs[event_index][selected_indices[-1]]))
    selected_indices.reverse()
    selected = [candidates_by_event[event_index][candidate_index] for event_index, candidate_index in enumerate(selected_indices)]

    transition_costs: list[float] = []
    for event_index in range(1, len(selected)):
        cost = transition_cost(
            selected[event_index - 1],
            selected[event_index],
            demos,
            intervals,
            weights=weights,
            contact_mask=contact_mask,
        )
        transition_costs.append(float(cost if np.isfinite(cost) else 0.0))

    report = {
        "status": "ok",
        "num_events": int(len(candidates_by_event)),
        "num_candidates": int(sum(len(group) for group in candidates_by_event)),
        "objective": float(np.max(scores[-1])),
        "selected_demo_ids": [int(candidate.demo_id) for candidate in selected],
        "unique_selected_demo_ids": sorted({int(candidate.demo_id) for candidate in selected}),
        "mean_local_score": float(np.mean([candidate.local_score for candidate in selected])) if selected else 0.0,
        "mean_transition_cost": float(np.mean(transition_costs)) if transition_costs else 0.0,
    }
    return selected, transition_costs, report


def _demo_row(demos: Any, demo_id: int) -> int:
    ids = np.asarray(demos.demo_ids, dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(ids == int(demo_id))
    if matches.size == 0:
        raise KeyError(f"Unknown demo id: {demo_id}")
    return int(matches[0])


def _velocity_at(values: np.ndarray, index: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    idx = int(np.clip(index, 0, arr.shape[0] - 1))
    if arr.shape[0] <= 1:
        return np.zeros_like(arr[idx])
    if idx == 0:
        return arr[1] - arr[0]
    return arr[idx] - arr[idx - 1]


def _l2(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr))
