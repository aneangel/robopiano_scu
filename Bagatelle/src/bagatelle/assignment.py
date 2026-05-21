from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


NUM_FINGERS = 10
FINGER_ORDER = tuple(range(NUM_FINGERS))
LEFT_HAND_FINGERS = tuple(range(5))
RIGHT_HAND_FINGERS = tuple(range(5, 10))
_LEGACY_TIEBREAK_FINGER = 1e-9
_LEGACY_TIEBREAK_KEY = 1e-12


@dataclass(frozen=True)
class FingerAssignmentResult:
    active_keys: np.ndarray
    assigned_finger_indices: np.ndarray
    assigned_keys: np.ndarray
    assigned_key_positions: np.ndarray
    target_positions: np.ndarray
    unassigned_keys: np.ndarray
    cost_matrix: np.ndarray
    total_cost: float
    mean_cost: float
    strategy: str = "legacy_previous_pose"
    cost_components: dict[str, np.ndarray] | None = None
    candidate_rank: int = 0
    candidate_score: float | None = None

    @property
    def count(self) -> int:
        return int(self.assigned_keys.shape[0])

    def dense_key_by_finger(self, *, fill_value: int = -1) -> np.ndarray:
        out = np.full((NUM_FINGERS,), int(fill_value), dtype=np.int32)
        if self.count:
            out[self.assigned_finger_indices.astype(np.int64)] = self.assigned_keys.astype(np.int32)
        return out

    def dense_cost_by_finger(self) -> np.ndarray:
        out = np.full((NUM_FINGERS,), np.nan, dtype=np.float32)
        if self.count:
            for finger, key_position in zip(self.assigned_finger_indices, self.assigned_key_positions):
                out[int(finger)] = float(self.cost_matrix[int(finger), int(key_position)])
        return out

    def dense_targets_by_finger(self) -> np.ndarray:
        out = np.full((NUM_FINGERS, 3), np.nan, dtype=np.float32)
        if self.count:
            out[self.assigned_finger_indices.astype(np.int64)] = self.target_positions.astype(np.float32)
        return out

    def _cost_component_summary(self) -> dict[str, dict[str, float]] | None:
        if not self.cost_components:
            return None
        summary: dict[str, dict[str, float]] = {}
        for name, component in self.cost_components.items():
            values = np.asarray(component, dtype=np.float64)
            if values.size == 0:
                summary[name] = {"mean": 0.0, "max": 0.0}
                continue
            summary[name] = {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "active_keys": self.active_keys.astype(int).tolist(),
            "assigned_finger_indices": self.assigned_finger_indices.astype(int).tolist(),
            "assigned_keys": self.assigned_keys.astype(int).tolist(),
            "unassigned_keys": self.unassigned_keys.astype(int).tolist(),
            "total_cost": float(self.total_cost),
            "mean_cost": float(self.mean_cost),
            "strategy": str(self.strategy),
            "candidate_rank": int(self.candidate_rank),
            "candidate_score": None if self.candidate_score is None else float(self.candidate_score),
        }
        cost_components = self._cost_component_summary()
        if cost_components is not None:
            payload["cost_components"] = cost_components
        return payload


@dataclass(frozen=True)
class FingerAssignmentCandidate:
    result: FingerAssignmentResult
    base_cost: float
    rank: int


def is_left_finger(finger_index: int) -> bool:
    return int(finger_index) in LEFT_HAND_FINGERS


def is_right_finger(finger_index: int) -> bool:
    return int(finger_index) in RIGHT_HAND_FINGERS


def finger_hand(finger_index: int) -> str:
    return "left" if is_left_finger(finger_index) else "right"


def is_black_key(key_index: int) -> bool:
    # Assumes piano key 0 corresponds to A0 (MIDI 21). Keep this isolated because
    # RoboPianist key indexing should be validated before relying on black-key costs.
    pitch_class = (int(key_index) + 9) % 12
    return pitch_class in {1, 3, 6, 8, 10}


def hand_zone_penalty(finger_index: int, key_index: int, config: object | None) -> float:
    middle_key = int(getattr(config, "assignment_middle_key", 44))
    wrong_hand_penalty = float(getattr(config, "assignment_wrong_hand_penalty", 1.0))
    hard_split = bool(getattr(config, "assignment_hard_hand_split", False))
    on_low_side = int(key_index) < middle_key
    if is_left_finger(finger_index) and not on_low_side:
        return 1e6 if hard_split else wrong_hand_penalty
    if is_right_finger(finger_index) and on_low_side:
        return 1e6 if hard_split else wrong_hand_penalty
    return 0.0


def finger_zone_penalty(
    finger_index: int,
    key_index: int,
    active_keys: np.ndarray,
    config: object | None,
) -> float:
    keys = np.asarray(active_keys, dtype=np.int32)
    if keys.size <= 1:
        return 0.0
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    position = int(np.searchsorted(sorted_keys, int(key_index), side="left"))
    normalized = float(position) / float(max(sorted_keys.size - 1, 1))
    if is_left_finger(finger_index):
        target = float(finger_index) / 4.0
    else:
        target = float(finger_index - 5) / 4.0
    return abs(normalized - target)


def reach_penalty(finger_index: int, distance: float, config: object | None) -> float:
    del finger_index
    soft_limit = max(float(getattr(config, "assignment_reach_soft_limit", 0.20)), 1e-6)
    return max(float(distance) - soft_limit, 0.0) / soft_limit


def normalize_active_keys_and_targets(
    raw_keys: np.ndarray,
    key_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(raw_keys, dtype=np.int32).reshape(-1)
    targets = np.asarray(key_targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"key_targets must have shape [N, 3] or [88, 3], got {targets.shape}")
    if targets.shape[0] >= 88:
        unique_keys = np.unique(keys.astype(np.int32))
        unique_keys.sort()
        if np.any(unique_keys < 0) or np.any(unique_keys >= targets.shape[0]):
            raise ValueError("active_keys contains an index outside key_targets")
        return unique_keys.astype(np.int32), np.ascontiguousarray(targets[unique_keys], dtype=np.float32)
    if targets.shape[0] != keys.shape[0]:
        raise ValueError(
            f"key_targets has {targets.shape[0]} rows but active_keys has {keys.shape[0]} entries"
        )
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order].astype(np.int32)
    sorted_targets = targets[order].astype(np.float32)
    if sorted_keys.size == 0:
        return sorted_keys, sorted_targets
    keep = np.concatenate([np.asarray([True]), sorted_keys[1:] != sorted_keys[:-1]])
    return sorted_keys[keep].astype(np.int32), np.ascontiguousarray(sorted_targets[keep], dtype=np.float32)


def _sorted_unique_keys_and_targets(raw_keys: np.ndarray, key_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return normalize_active_keys_and_targets(raw_keys, key_targets)


def build_legacy_distance_cost(
    fingers: np.ndarray,
    targets: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    del keys
    diff = np.asarray(fingers, dtype=np.float32)[:, None, :] - np.asarray(targets, dtype=np.float32)[None, :, :]
    cost = np.linalg.norm(diff, axis=2).astype(np.float64)
    cost += np.arange(NUM_FINGERS, dtype=np.float64)[:, None] * _LEGACY_TIEBREAK_FINGER
    cost += np.arange(targets.shape[0], dtype=np.float64)[None, :] * _LEGACY_TIEBREAK_KEY
    return cost


def _cost_bias_for_keys(cost_bias: np.ndarray, keys: np.ndarray, *, key_count: int) -> np.ndarray:
    bias = np.asarray(cost_bias, dtype=np.float32)
    if bias.ndim != 2:
        raise ValueError(f"cost_bias must have shape [10, K] or [10, 88], got {bias.shape}")
    if bias.shape == (NUM_FINGERS, key_count):
        return bias.astype(np.float64, copy=False)
    if bias.shape[0] != NUM_FINGERS:
        raise ValueError(f"cost_bias must have 10 rows, got {bias.shape}")
    if bias.shape[1] >= 88:
        return bias[:, keys.astype(np.int64)].astype(np.float64, copy=False)
    raise ValueError(f"cost_bias must have shape [10, {key_count}] or [10, 88], got {bias.shape}")


def _resolve_previous_assignment(
    previous_assignment: np.ndarray | None = None,
    previous_assigned_keys: np.ndarray | None = None,
) -> np.ndarray | None:
    dense = previous_assignment if previous_assignment is not None else previous_assigned_keys
    if dense is None:
        return None
    flat = np.asarray(dense, dtype=np.int32).reshape(-1)
    if flat.shape != (NUM_FINGERS,):
        raise ValueError(f"previous_assignment must have shape [10], got {flat.shape}")
    return flat


def _assignment_next_fingertips(
    previous_fingertips: np.ndarray,
    assignment: FingerAssignmentResult,
) -> np.ndarray:
    next_fingertips = np.asarray(previous_fingertips, dtype=np.float32).copy()
    if assignment.count:
        next_fingertips[assignment.assigned_finger_indices.astype(np.int64)] = (
            assignment.target_positions.astype(np.float32)
        )
    return next_fingertips


def _assignment_next_keys(
    previous_assigned_keys: np.ndarray,
    assignment: FingerAssignmentResult,
) -> np.ndarray:
    next_keys = np.full((NUM_FINGERS,), -1, dtype=np.int32)
    if assignment.count:
        next_keys[assignment.assigned_finger_indices.astype(np.int64)] = assignment.assigned_keys.astype(np.int32)
    return next_keys


def _hand_fingers(finger_index: int) -> tuple[int, ...]:
    return LEFT_HAND_FINGERS if int(finger_index) < 5 else RIGHT_HAND_FINGERS


def _same_hand_sorted_by_y(previous_fingertips: np.ndarray, fingers: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    hand = np.asarray(fingers, dtype=np.int32)
    order = np.argsort(previous_fingertips[hand, 1], kind="stable")
    return hand, hand[order]


def _continuity_cost_matrix(
    *,
    keys: np.ndarray,
    targets: np.ndarray,
    previous_fingertips: np.ndarray,
    previous_assignment: np.ndarray | None,
    config: object | None,
) -> np.ndarray:
    cost = np.zeros((NUM_FINGERS, keys.shape[0]), dtype=np.float64)
    if keys.size == 0 or config is None:
        return cost

    previous_keys = _resolve_previous_assignment(previous_assignment)
    distance_weight = float(getattr(config, "distance_weight", 1.0))
    same_finger_bonus = float(getattr(config, "same_finger_bonus", 0.0))
    reassignment_penalty = float(getattr(config, "reassignment_penalty", 0.0))
    finger_crossing_penalty = float(getattr(config, "finger_crossing_penalty", 0.0))
    wrong_hand_penalty = float(getattr(config, "wrong_hand_penalty", 0.0))
    large_jump_penalty = float(getattr(config, "large_jump_penalty", 0.0))
    same_key_same_finger_bonus = float(getattr(config, "same_key_same_finger_bonus", 0.0))
    split_key = int(getattr(config, "wrong_hand_split_key", 44))
    jump_distance = max(float(getattr(config, "large_jump_distance_m", 0.06)), 1e-6)
    crossing_slack = max(float(getattr(config, "finger_crossing_slack_m", 0.005)), 0.0)

    if distance_weight != 1.0:
        cost += 0.0

    previous_key_owner: dict[int, int] = {}
    if previous_keys is not None:
        for finger_index, key in enumerate(previous_keys.tolist()):
            if int(key) >= 0:
                previous_key_owner[int(key)] = int(finger_index)

    delta_xy = np.abs(previous_fingertips[:, None, 1] - targets[None, :, 1]).astype(np.float64)

    for finger_index in range(NUM_FINGERS):
        if previous_keys is not None and int(previous_keys[finger_index]) >= 0:
            if same_finger_bonus > 0.0:
                cost[finger_index, :] -= same_finger_bonus
            if same_key_same_finger_bonus > 0.0:
                mask = keys == int(previous_keys[finger_index])
                cost[finger_index, mask] -= same_key_same_finger_bonus

        if wrong_hand_penalty > 0.0:
            if finger_index < 5:
                wrong_hand_mask = keys > split_key
            else:
                wrong_hand_mask = keys < split_key
            cost[finger_index, wrong_hand_mask] += wrong_hand_penalty

        if large_jump_penalty > 0.0:
            excess = np.maximum(delta_xy[finger_index] - jump_distance, 0.0) / jump_distance
            cost[finger_index, :] += large_jump_penalty * excess

        if finger_crossing_penalty > 0.0:
            _, sorted_hand = _same_hand_sorted_by_y(previous_fingertips, _hand_fingers(finger_index))
            hand_order = sorted_hand.tolist()
            rank = hand_order.index(int(finger_index))
            target_y = targets[:, 1].astype(np.float64)
            lower = previous_fingertips[sorted_hand[:rank], 1].astype(np.float64)
            upper = previous_fingertips[sorted_hand[rank + 1 :], 1].astype(np.float64)
            lower_bound = np.max(lower) + crossing_slack if lower.size else -np.inf
            upper_bound = np.min(upper) - crossing_slack if upper.size else np.inf
            below = np.maximum(lower_bound - target_y, 0.0)
            above = np.maximum(target_y - upper_bound, 0.0)
            crossing_excess = (below + above) / max(jump_distance, 1e-6)
            cost[finger_index, :] += finger_crossing_penalty * crossing_excess

    if reassignment_penalty > 0.0 and previous_key_owner:
        for key_position, key in enumerate(keys.tolist()):
            owner = previous_key_owner.get(int(key))
            if owner is None:
                continue
            cost[:, key_position] += reassignment_penalty
            cost[int(owner), key_position] -= reassignment_penalty

    return cost


def build_composite_assignment_cost(
    keys: np.ndarray,
    targets: np.ndarray,
    previous_fingertips: np.ndarray,
    config: object | None,
    previous_assignment: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    previous = np.asarray(previous_fingertips, dtype=np.float32)
    diff = previous[:, None, :] - np.asarray(targets, dtype=np.float32)[None, :, :]
    distance = np.linalg.norm(diff, axis=2).astype(np.float64)

    components: dict[str, np.ndarray] = {
        "distance": distance,
        "hand_zone": np.zeros_like(distance),
        "finger_zone": np.zeros_like(distance),
        "hold": np.zeros_like(distance),
        "reach": np.zeros_like(distance),
        "black_key": np.zeros_like(distance),
        "deterministic_tie_break": np.zeros_like(distance),
    }
    previous_dense = _resolve_previous_assignment(previous_assignment)
    previous_key_owner: dict[int, int] = {}
    if previous_dense is not None:
        for finger_index, key in enumerate(previous_dense.tolist()):
            if int(key) >= 0:
                previous_key_owner[int(key)] = int(finger_index)

    for finger_index in range(NUM_FINGERS):
        for key_position, key in enumerate(np.asarray(keys, dtype=np.int32).tolist()):
            components["hand_zone"][finger_index, key_position] = hand_zone_penalty(finger_index, key, config)
            components["finger_zone"][finger_index, key_position] = finger_zone_penalty(
                finger_index,
                key,
                np.asarray(keys, dtype=np.int32),
                config,
            )
            components["reach"][finger_index, key_position] = reach_penalty(
                finger_index,
                float(distance[finger_index, key_position]),
                config,
            )
            components["black_key"][finger_index, key_position] = 1.0 if is_black_key(key) else 0.0
            components["deterministic_tie_break"][finger_index, key_position] = (
                float(finger_index) * _LEGACY_TIEBREAK_FINGER
                + float(key_position) * _LEGACY_TIEBREAK_KEY
            )
            owner = previous_key_owner.get(int(key))
            if owner is not None and owner != finger_index:
                components["hold"][finger_index, key_position] = 1.0

    if previous_dense is not None:
        for finger_index, key in enumerate(previous_dense.tolist()):
            if int(key) < 0:
                continue
            matches = np.where(np.asarray(keys, dtype=np.int32) == int(key))[0]
            if matches.size:
                components["hold"][finger_index, int(matches[0])] = 0.0

    cost = (
        float(getattr(config, "assignment_distance_weight", 1.0)) * components["distance"]
        + float(getattr(config, "assignment_hand_zone_weight", 0.0)) * components["hand_zone"]
        + float(getattr(config, "assignment_finger_zone_weight", 0.0)) * components["finger_zone"]
        + float(getattr(config, "assignment_hold_weight", 0.0)) * components["hold"]
        + float(getattr(config, "assignment_reach_weight", 0.0)) * components["reach"]
        + float(getattr(config, "assignment_black_key_weight", 0.0)) * components["black_key"]
        + components["deterministic_tie_break"]
    )
    return cost.astype(np.float64), components


def _lookahead_pair_penalty(
    *,
    finger_index: int,
    current_key: int,
    current_target: np.ndarray,
    previous_fingertips: np.ndarray,
    future_active_keys: Sequence[np.ndarray],
    future_key_targets: Sequence[np.ndarray],
    previous_assignment: np.ndarray | None,
    config: object | None,
    depth: int,
) -> float:
    if depth <= 0 or len(future_active_keys) == 0:
        return 0.0
    hypothetical = np.asarray(previous_fingertips, dtype=np.float32).copy()
    hypothetical[int(finger_index)] = np.asarray(current_target, dtype=np.float32)
    hypothetical_keys = None
    if previous_assignment is not None:
        hypothetical_keys = np.asarray(previous_assignment, dtype=np.int32).copy()
        hypothetical_keys[int(finger_index)] = int(current_key)
    future = assign_fingers_previous_pose_lookahead(
        future_active_keys[0],
        hypothetical,
        future_key_targets[0],
        future_active_keys=future_active_keys[1:],
        future_key_targets=future_key_targets[1:],
        previous_assignment=hypothetical_keys,
        lookahead_steps=depth - 1,
        config=config,
    )
    return float(future.total_cost)


def _lookahead_bias(
    keys: np.ndarray,
    targets: np.ndarray,
    previous_fingertips: np.ndarray,
    future_active_keys: Sequence[np.ndarray],
    future_key_targets: Sequence[np.ndarray],
    previous_assignment: np.ndarray | None,
    config: object | None,
    depth: int,
) -> np.ndarray:
    bias = np.zeros((NUM_FINGERS, targets.shape[0]), dtype=np.float64)
    if depth <= 0 or targets.shape[0] == 0 or len(future_active_keys) == 0:
        return bias
    for finger_index in range(NUM_FINGERS):
        for key_position in range(targets.shape[0]):
            bias[finger_index, key_position] = _lookahead_pair_penalty(
                finger_index=finger_index,
                current_key=int(keys[key_position]),
                current_target=targets[key_position],
                previous_fingertips=previous_fingertips,
                future_active_keys=future_active_keys,
                future_key_targets=future_key_targets,
                previous_assignment=previous_assignment,
                config=config,
                depth=depth,
            )
    return bias


def _empty_result(
    active_keys: np.ndarray,
    cost_matrix: np.ndarray,
    *,
    strategy: str = "legacy_previous_pose",
    cost_components: dict[str, np.ndarray] | None = None,
    candidate_rank: int = 0,
    candidate_score: float | None = None,
) -> FingerAssignmentResult:
    return FingerAssignmentResult(
        active_keys=active_keys.astype(np.int32),
        assigned_finger_indices=np.zeros((0,), dtype=np.int32),
        assigned_keys=np.zeros((0,), dtype=np.int32),
        assigned_key_positions=np.zeros((0,), dtype=np.int32),
        target_positions=np.zeros((0, 3), dtype=np.float32),
        unassigned_keys=np.zeros((0,), dtype=np.int32),
        cost_matrix=cost_matrix.astype(np.float32, copy=False),
        total_cost=0.0,
        mean_cost=0.0,
        strategy=strategy,
        cost_components=cost_components,
        candidate_rank=candidate_rank,
        candidate_score=candidate_score,
    )


def _result_from_cost(
    *,
    keys: np.ndarray,
    targets: np.ndarray,
    cost: np.ndarray,
    strategy: str,
    cost_components: dict[str, np.ndarray] | None = None,
    candidate_rank: int = 0,
    candidate_score: float | None = None,
) -> FingerAssignmentResult:
    if keys.size == 0:
        return _empty_result(
            keys,
            np.zeros((NUM_FINGERS, 0), dtype=np.float32),
            strategy=strategy,
            cost_components=cost_components,
            candidate_rank=candidate_rank,
            candidate_score=candidate_score,
        )

    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.lexsort((keys[col_ind], row_ind))
    row_ind = row_ind[order].astype(np.int32)
    col_ind = col_ind[order].astype(np.int32)

    assigned_keys = keys[col_ind].astype(np.int32)
    assigned_targets = targets[col_ind].astype(np.float32)
    unassigned_mask = np.ones((keys.shape[0],), dtype=bool)
    unassigned_mask[col_ind] = False
    selected_costs = cost[row_ind, col_ind].astype(np.float32)

    stored_components = cost_components if bool(getattr(cost_components, "items", None)) else None
    return FingerAssignmentResult(
        active_keys=keys.astype(np.int32),
        assigned_finger_indices=row_ind,
        assigned_keys=assigned_keys,
        assigned_key_positions=col_ind,
        target_positions=assigned_targets,
        unassigned_keys=keys[unassigned_mask].astype(np.int32),
        cost_matrix=cost.astype(np.float32),
        total_cost=float(np.sum(selected_costs)),
        mean_cost=float(np.mean(selected_costs)) if selected_costs.size else 0.0,
        strategy=strategy,
        cost_components=stored_components,
        candidate_rank=candidate_rank,
        candidate_score=candidate_score,
    )


def _strategy_name(config: object | None) -> str:
    return str(getattr(config, "assignment_strategy", "legacy_previous_pose"))


def assignment_crossing_penalty(
    assigned_finger_indices: np.ndarray,
    assigned_keys: np.ndarray,
    config: object | None = None,
) -> float:
    del config
    fingers = np.asarray(assigned_finger_indices, dtype=np.int32)
    keys = np.asarray(assigned_keys, dtype=np.int32)
    penalty = 0.0
    for hand in (LEFT_HAND_FINGERS, RIGHT_HAND_FINGERS):
        hand_mask = np.isin(fingers, np.asarray(hand, dtype=np.int32))
        hand_fingers = fingers[hand_mask]
        hand_keys = keys[hand_mask]
        if hand_fingers.size <= 1:
            continue
        order = np.argsort(hand_fingers, kind="stable")
        ordered_keys = hand_keys[order]
        for left_index in range(ordered_keys.size):
            for right_index in range(left_index + 1, ordered_keys.size):
                if int(ordered_keys[left_index]) > int(ordered_keys[right_index]):
                    penalty += 1.0
    return float(penalty)


def generate_assignment_candidates(
    active_keys: np.ndarray,
    previous_fingertips: np.ndarray,
    key_targets: np.ndarray,
    config: object | None = None,
    *,
    max_candidates: int | None = None,
    previous_assignment: np.ndarray | None = None,
    previous_assigned_keys: np.ndarray | None = None,
) -> list[FingerAssignmentCandidate]:
    raw_keys = np.asarray(active_keys, dtype=np.int32).reshape(-1)
    keys, targets = normalize_active_keys_and_targets(raw_keys, key_targets)
    if np.any(keys < 0) or np.any(keys >= 88):
        raise ValueError(f"active_keys must be piano key indices in [0, 87], got {keys}")

    fingers = np.asarray(previous_fingertips, dtype=np.float32)
    if fingers.shape != (NUM_FINGERS, 3):
        raise ValueError(f"previous_fingertips must have shape [10, 3], got {fingers.shape}")

    strategy = _strategy_name(config)
    dense_previous = _resolve_previous_assignment(previous_assignment, previous_assigned_keys)
    top_k = int(max_candidates if max_candidates is not None else getattr(config, "assignment_top_k", 1))
    top_k = max(top_k, 1)

    if strategy == "legacy_previous_pose":
        legacy = assign_fingers_previous_pose(
            keys,
            fingers,
            targets,
            config,
            previous_assignment=dense_previous,
        )
        return [FingerAssignmentCandidate(result=legacy, base_cost=float(legacy.total_cost), rank=0)]

    if strategy not in {"composite_cost", "ik_aware_topk"}:
        raise ValueError(f"Unknown Bagatelle assignment_strategy: {strategy}")

    base_cost, cost_components = build_composite_assignment_cost(
        keys,
        targets,
        fingers,
        config,
        previous_assignment=dense_previous,
    )
    stored_components = cost_components if bool(getattr(config, "assignment_store_cost_components", True)) else None
    base_result = _result_from_cost(
        keys=keys,
        targets=targets,
        cost=base_cost,
        strategy=strategy,
        cost_components=stored_components,
        candidate_rank=0,
    )
    candidates: list[FingerAssignmentCandidate] = [
        FingerAssignmentCandidate(result=base_result, base_cost=float(base_result.total_cost), rank=0)
    ]
    if top_k <= 1 or base_result.count == 0:
        return candidates

    seen = {
        (
            tuple(base_result.assigned_finger_indices.astype(int).tolist()),
            tuple(base_result.assigned_keys.astype(int).tolist()),
        )
    }
    perturb_scale = float(getattr(config, "assignment_top_k_extra_penalty", 1e-4))
    perturb_targets = list(zip(base_result.assigned_finger_indices.tolist(), base_result.assigned_key_positions.tolist()))
    candidate_rank = 1
    for finger_index, key_position in perturb_targets:
        if len(candidates) >= top_k:
            break
        perturbed = np.array(base_cost, copy=True)
        perturbed[int(finger_index), int(key_position)] += perturb_scale * float(candidate_rank)
        result = _result_from_cost(
            keys=keys,
            targets=targets,
            cost=perturbed,
            strategy=strategy,
            cost_components=stored_components,
            candidate_rank=candidate_rank,
        )
        signature = (
            tuple(result.assigned_finger_indices.astype(int).tolist()),
            tuple(result.assigned_keys.astype(int).tolist()),
        )
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(FingerAssignmentCandidate(result=result, base_cost=float(result.total_cost), rank=candidate_rank))
        candidate_rank += 1

    candidates.sort(
        key=lambda candidate: (
            float(candidate.base_cost),
            tuple(candidate.result.assigned_finger_indices.astype(int).tolist()),
            tuple(candidate.result.assigned_keys.astype(int).tolist()),
        )
    )
    ranked: list[FingerAssignmentCandidate] = []
    for rank, candidate in enumerate(candidates):
        ranked_result = FingerAssignmentResult(
            active_keys=candidate.result.active_keys,
            assigned_finger_indices=candidate.result.assigned_finger_indices,
            assigned_keys=candidate.result.assigned_keys,
            assigned_key_positions=candidate.result.assigned_key_positions,
            target_positions=candidate.result.target_positions,
            unassigned_keys=candidate.result.unassigned_keys,
            cost_matrix=candidate.result.cost_matrix,
            total_cost=candidate.result.total_cost,
            mean_cost=candidate.result.mean_cost,
            strategy=candidate.result.strategy,
            cost_components=candidate.result.cost_components,
            candidate_rank=rank,
            candidate_score=candidate.result.candidate_score,
        )
        ranked.append(FingerAssignmentCandidate(result=ranked_result, base_cost=float(candidate.base_cost), rank=rank))
    return ranked[:top_k]


def assign_fingers_previous_pose_lookahead(
    active_keys: np.ndarray,
    previous_fingertips: np.ndarray,
    key_targets: np.ndarray,
    *,
    future_active_keys: Sequence[np.ndarray] = (),
    future_key_targets: Sequence[np.ndarray] = (),
    previous_assignment: np.ndarray | None = None,
    previous_assigned_keys: np.ndarray | None = None,
    lookahead_steps: int = 1,
    config: object | None = None,
    lookahead_weight: float = 1.0,
) -> FingerAssignmentResult:
    raw_keys = np.asarray(active_keys, dtype=np.int32).reshape(-1)
    keys, targets = normalize_active_keys_and_targets(raw_keys, key_targets)
    if int(lookahead_steps) <= 0 or len(future_active_keys) == 0 or keys.size == 0:
        return assign_fingers_previous_pose(
            active_keys,
            previous_fingertips,
            key_targets,
            config,
            previous_assignment=previous_assignment,
            previous_assigned_keys=previous_assigned_keys,
        )
    depth = min(int(lookahead_steps), len(future_active_keys), len(future_key_targets))
    dense_previous = _resolve_previous_assignment(previous_assignment, previous_assigned_keys)
    bias = _lookahead_bias(
        keys,
        targets,
        np.asarray(previous_fingertips, dtype=np.float32),
        future_active_keys[:depth],
        future_key_targets[:depth],
        dense_previous,
        config,
        depth,
    )
    return assign_fingers_previous_pose(
        keys,
        previous_fingertips,
        targets,
        config,
        previous_assignment=dense_previous,
        cost_bias=bias,
        cost_bias_alpha=float(lookahead_weight),
    )


def assign_fingers_sequence_lookahead(
    active_key_sequence: Sequence[np.ndarray],
    initial_fingertips: np.ndarray,
    key_target_sequence: Sequence[np.ndarray],
    *,
    lookahead_steps: int = 1,
    config: object | None = None,
    lookahead_weight: float = 1.0,
) -> list[FingerAssignmentResult]:
    if len(active_key_sequence) != len(key_target_sequence):
        raise ValueError("active_key_sequence and key_target_sequence must have the same length")
    previous = np.asarray(initial_fingertips, dtype=np.float32).copy()
    results: list[FingerAssignmentResult] = []
    previous_keys = np.full((NUM_FINGERS,), -1, dtype=np.int32)
    for index, (active_keys, key_targets) in enumerate(zip(active_key_sequence, key_target_sequence)):
        result = assign_fingers_previous_pose_lookahead(
            active_keys,
            previous,
            key_targets,
            future_active_keys=active_key_sequence[index + 1 :],
            future_key_targets=key_target_sequence[index + 1 :],
            previous_assignment=previous_keys,
            lookahead_steps=lookahead_steps,
            config=config,
            lookahead_weight=lookahead_weight,
        )
        results.append(result)
        previous = _assignment_next_fingertips(previous, result)
        previous_keys = _assignment_next_keys(previous_keys, result)
    return results


def assign_fingers_previous_pose(
    active_keys: np.ndarray,
    previous_fingertips: np.ndarray,
    key_targets: np.ndarray,
    config: object | None = None,
    *,
    previous_assignment: np.ndarray | None = None,
    previous_assigned_keys: np.ndarray | None = None,
    cost_bias: np.ndarray | None = None,
    cost_bias_alpha: float | None = None,
) -> FingerAssignmentResult:
    raw_keys = np.asarray(active_keys, dtype=np.int32).reshape(-1)
    keys, targets = normalize_active_keys_and_targets(raw_keys, key_targets)
    if np.any(keys < 0) or np.any(keys >= 88):
        raise ValueError(f"active_keys must be piano key indices in [0, 87], got {keys}")

    fingers = np.asarray(previous_fingertips, dtype=np.float32)
    if fingers.shape != (NUM_FINGERS, 3):
        raise ValueError(f"previous_fingertips must have shape [10, 3], got {fingers.shape}")

    strategy = _strategy_name(config)
    dense_previous = _resolve_previous_assignment(previous_assignment, previous_assigned_keys)
    if keys.size == 0:
        return _empty_result(keys, np.zeros((NUM_FINGERS, 0), dtype=np.float32), strategy=strategy)

    if strategy == "legacy_previous_pose":
        distance_cost = build_legacy_distance_cost(fingers, targets, keys)
        cost = float(getattr(config, "distance_weight", 1.0)) * distance_cost
        cost += _continuity_cost_matrix(
            keys=keys,
            targets=targets,
            previous_fingertips=fingers,
            previous_assignment=dense_previous,
            config=config,
        )
        if cost_bias is not None:
            bias = _cost_bias_for_keys(cost_bias, keys, key_count=keys.shape[0])
            alpha = float(getattr(config, "cost_bias_alpha", 1.0) if cost_bias_alpha is None else cost_bias_alpha)
            cost = cost + alpha * bias.astype(np.float64)
        return _result_from_cost(keys=keys, targets=targets, cost=cost, strategy=strategy)

    if strategy in {"composite_cost", "ik_aware_topk"}:
        cost, cost_components = build_composite_assignment_cost(
            keys,
            targets,
            fingers,
            config,
            previous_assignment=dense_previous,
        )
        if cost_bias is not None:
            bias = _cost_bias_for_keys(cost_bias, keys, key_count=keys.shape[0])
            alpha = float(getattr(config, "cost_bias_alpha", 1.0) if cost_bias_alpha is None else cost_bias_alpha)
            cost = cost + alpha * bias.astype(np.float64)
        stored_components = cost_components if bool(getattr(config, "assignment_store_cost_components", True)) else None
        return _result_from_cost(
            keys=keys,
            targets=targets,
            cost=cost,
            strategy=strategy,
            cost_components=stored_components,
        )

    raise ValueError(f"Unknown Bagatelle assignment_strategy: {strategy}")
