from __future__ import annotations

from typing import Any

import numpy as np

from nocturne.candidate_outcomes import correctness_severity, strict_correctness_severity
from nocturne.schema import SegmentCandidate, StitchConfig, TransitionWeights
from nocturne.viterbi import transition_cost


def repair_selected_path(
    selected: list[SegmentCandidate],
    candidates_by_event: list[list[SegmentCandidate]],
    demos: Any,
    intervals: list[tuple[int, int]],
    *,
    config: StitchConfig,
    weights: TransitionWeights | None = None,
    contact_mask: np.ndarray | None = None,
) -> tuple[list[SegmentCandidate], list[float], dict[str, Any]]:
    """Greedily swap local candidates when doing so removes avoidable correctness errors."""
    if not selected or not bool(config.repair_enabled) or str(config.objective_mode) == "legacy":
        costs = _path_transition_costs(selected, demos, intervals, weights=weights, contact_mask=contact_mask)
        return selected, costs, {"status": "skipped", "num_swaps": 0}

    path = list(selected)
    swaps: list[dict[str, Any]] = []
    max_passes = max(int(config.repair_max_passes), 0)
    for pass_index in range(max_passes):
        changed = False
        for event_index in range(len(path)):
            replacement, reason = _best_replacement(
                path,
                event_index,
                candidates_by_event[event_index],
                demos,
                intervals,
                weights=weights,
                contact_mask=contact_mask,
                transition_margin=float(config.repair_transition_margin),
                strict=bool(str(config.objective_mode) == "strict"),
            )
            if replacement is None:
                continue
            old = path[event_index]
            path[event_index] = replacement
            changed = True
            swaps.append(
                {
                    "pass_index": int(pass_index),
                    "event_index": int(event_index),
                    "old_demo_id": int(old.demo_id),
                    "new_demo_id": int(replacement.demo_id),
                    "old_segment_id": int(old.segment_id),
                    "new_segment_id": int(replacement.segment_id),
                    **reason,
                }
            )
        if not changed:
            break

    costs = _path_transition_costs(path, demos, intervals, weights=weights, contact_mask=contact_mask)
    return path, costs, {"status": "ok", "num_swaps": int(len(swaps)), "swaps": swaps[:200]}


def _best_replacement(
    path: list[SegmentCandidate],
    event_index: int,
    candidates: list[SegmentCandidate],
    demos: Any,
    intervals: list[tuple[int, int]],
    *,
    weights: TransitionWeights | None,
    contact_mask: np.ndarray | None,
    transition_margin: float,
    strict: bool,
) -> tuple[SegmentCandidate | None, dict[str, Any]]:
    current = path[event_index]
    current_severity = _severity(current, strict=strict)
    current_cost = _neighbor_transition_cost(path, event_index, demos, intervals, weights=weights, contact_mask=contact_mask)
    best: SegmentCandidate | None = None
    best_key: tuple[Any, ...] | None = None
    best_reason: dict[str, Any] = {}
    for candidate in candidates:
        if int(candidate.segment_id) == int(current.segment_id):
            continue
        severity = _severity(candidate, strict=strict)
        if severity >= current_severity:
            continue
        trial = list(path)
        trial[event_index] = candidate
        cost = _neighbor_transition_cost(trial, event_index, demos, intervals, weights=weights, contact_mask=contact_mask)
        if not np.isfinite(cost):
            continue
        if np.isfinite(current_cost) and cost > current_cost + float(transition_margin):
            continue
        sort_key = (severity, float(cost), -float(candidate.local_score), int(candidate.demo_id))
        if best_key is None or sort_key < best_key:
            best = candidate
            best_key = sort_key
            best_reason = {
                "old_severity": list(current_severity),
                "new_severity": list(severity),
                "old_neighbor_transition_cost": float(current_cost if np.isfinite(current_cost) else 0.0),
                "new_neighbor_transition_cost": float(cost),
            }
    return best, best_reason


def _severity(candidate: SegmentCandidate, *, strict: bool) -> tuple[int, ...]:
    return strict_correctness_severity(candidate) if strict else correctness_severity(candidate)


def _neighbor_transition_cost(
    path: list[SegmentCandidate],
    event_index: int,
    demos: Any,
    intervals: list[tuple[int, int]],
    *,
    weights: TransitionWeights | None,
    contact_mask: np.ndarray | None,
) -> float:
    cost = 0.0
    if event_index > 0:
        left = transition_cost(
            path[event_index - 1],
            path[event_index],
            demos,
            intervals,
            weights=weights,
            contact_mask=contact_mask,
        )
        if not np.isfinite(left):
            return float("inf")
        cost += float(left)
    if event_index + 1 < len(path):
        right = transition_cost(
            path[event_index],
            path[event_index + 1],
            demos,
            intervals,
            weights=weights,
            contact_mask=contact_mask,
        )
        if not np.isfinite(right):
            return float("inf")
        cost += float(right)
    return float(cost)


def _path_transition_costs(
    path: list[SegmentCandidate],
    demos: Any,
    intervals: list[tuple[int, int]],
    *,
    weights: TransitionWeights | None,
    contact_mask: np.ndarray | None,
) -> list[float]:
    costs = []
    for event_index in range(1, len(path)):
        cost = transition_cost(
            path[event_index - 1],
            path[event_index],
            demos,
            intervals,
            weights=weights,
            contact_mask=contact_mask,
        )
        costs.append(float(cost if np.isfinite(cost) else 0.0))
    return costs
