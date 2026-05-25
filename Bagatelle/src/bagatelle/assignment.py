from __future__ import annotations

from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class _AssignmentBeamState:
    previous_fingertips: np.ndarray
    previous_assignment: np.ndarray
    cumulative_cost: float
    results: tuple[FingerAssignmentResult, ...]


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
    return hand_zone_penalty_for_split(
        finger_index,
        key_index,
        middle_key=middle_key,
        wrong_hand_penalty=wrong_hand_penalty,
        hard_split=hard_split,
    )


def hand_zone_penalty_for_split(
    finger_index: int,
    key_index: int,
    *,
    middle_key: int,
    wrong_hand_penalty: float,
    hard_split: bool,
) -> float:
    on_low_side = int(key_index) < middle_key
    if is_left_finger(finger_index) and not on_low_side:
        return 1e6 if hard_split else wrong_hand_penalty
    if is_right_finger(finger_index) and on_low_side:
        return 1e6 if hard_split else wrong_hand_penalty
    return 0.0


def dynamic_hand_split_key(keys: np.ndarray, config: object | None, *, default_split_key: int) -> int:
    if config is None or not bool(getattr(config, "assignment_dynamic_hand_split", False)):
        return int(default_split_key)
    values = np.unique(np.asarray(keys, dtype=np.int32).reshape(-1))
    values.sort()
    min_keys = max(int(getattr(config, "assignment_dynamic_hand_split_min_keys", 3)), 2)
    min_span = max(int(getattr(config, "assignment_dynamic_hand_split_min_span", 12)), 1)
    if values.size < min_keys or int(values[-1] - values[0]) < min_span:
        return int(default_split_key)
    gaps = np.diff(values)
    if gaps.size == 0:
        return int(default_split_key)
    max_gap = int(np.max(gaps))
    candidates = np.flatnonzero(gaps == max_gap)
    center = (values.size - 1) / 2.0
    gap_index = int(
        sorted(
            candidates.tolist(),
            key=lambda index: (abs((float(index) + 0.5) - center), -int(index)),
        )[0]
    )
    low = int(values[gap_index])
    high = int(values[gap_index + 1])
    if high <= low:
        return int(default_split_key)
    return int((low + high) // 2)


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
        target = float(4 - finger_index) / 4.0
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
    split_key = dynamic_hand_split_key(
        keys,
        config,
        default_split_key=int(getattr(config, "wrong_hand_split_key", 44)),
    )
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
            middle_key = dynamic_hand_split_key(
                keys,
                config,
                default_split_key=int(getattr(config, "assignment_middle_key", 44)),
            )
            components["hand_zone"][finger_index, key_position] = hand_zone_penalty_for_split(
                finger_index,
                key,
                middle_key=middle_key,
                wrong_hand_penalty=float(getattr(config, "assignment_wrong_hand_penalty", 1.0)),
                hard_split=bool(getattr(config, "assignment_hard_hand_split", False)),
            )
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
        if hand == LEFT_HAND_FINGERS:
            ranks = np.asarray([4 - int(finger) for finger in hand_fingers], dtype=np.int32)
        else:
            ranks = np.asarray([int(finger) - 5 for finger in hand_fingers], dtype=np.int32)
        order = np.argsort(ranks, kind="stable")
        ordered_keys = hand_keys[order]
        for left_index in range(ordered_keys.size):
            for right_index in range(left_index + 1, ordered_keys.size):
                if int(ordered_keys[left_index]) > int(ordered_keys[right_index]):
                    penalty += 1.0
    return float(penalty)


def _candidate_signature(result: FingerAssignmentResult) -> tuple[int, ...]:
    return tuple(result.dense_key_by_finger().astype(int).tolist())


def _rank_candidate_results(results: list[FingerAssignmentCandidate], top_k: int) -> list[FingerAssignmentCandidate]:
    results.sort(
        key=lambda candidate: (
            float(candidate.base_cost),
            tuple(candidate.result.assigned_finger_indices.astype(int).tolist()),
            tuple(candidate.result.assigned_keys.astype(int).tolist()),
        )
    )
    ranked: list[FingerAssignmentCandidate] = []
    for rank, candidate in enumerate(results[:top_k]):
        ranked_result = replace(candidate.result, candidate_rank=rank)
        ranked.append(FingerAssignmentCandidate(result=ranked_result, base_cost=float(candidate.base_cost), rank=rank))
    return ranked


def _crossed_selected_pairs(result: FingerAssignmentResult) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    fingers = result.assigned_finger_indices.astype(np.int32)
    keys = result.assigned_keys.astype(np.int32)
    key_positions = result.assigned_key_positions.astype(np.int32)
    for hand in (LEFT_HAND_FINGERS, RIGHT_HAND_FINGERS):
        hand_mask = np.isin(fingers, np.asarray(hand, dtype=np.int32))
        hand_fingers = fingers[hand_mask]
        hand_keys = keys[hand_mask]
        hand_positions = key_positions[hand_mask]
        if hand == LEFT_HAND_FINGERS:
            ranks = np.asarray([4 - int(finger) for finger in hand_fingers], dtype=np.int32)
        else:
            ranks = np.asarray([int(finger) - 5 for finger in hand_fingers], dtype=np.int32)
        order = np.argsort(ranks, kind="stable")
        ordered_fingers = hand_fingers[order]
        ordered_keys = hand_keys[order]
        ordered_positions = hand_positions[order]
        for left_index in range(ordered_fingers.size):
            for right_index in range(left_index + 1, ordered_fingers.size):
                if int(ordered_keys[left_index]) > int(ordered_keys[right_index]):
                    pairs.append((int(ordered_fingers[left_index]), int(ordered_positions[left_index])))
                    pairs.append((int(ordered_fingers[right_index]), int(ordered_positions[right_index])))
    return pairs


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

    if strategy not in {"composite_cost", "ik_aware_topk", "sequence_beam"}:
        raise ValueError(f"Unknown Bagatelle assignment_strategy: {strategy}")

    base_cost, cost_components = build_composite_assignment_cost(
        keys,
        targets,
        fingers,
        config,
        previous_assignment=dense_previous,
    )
    stored_components = cost_components if bool(getattr(config, "assignment_store_cost_components", True)) else None
    candidates: list[FingerAssignmentCandidate] = []
    seen: set[tuple[int, ...]] = set()

    def add_candidate(cost: np.ndarray, rank_hint: int) -> None:
        result = _result_from_cost(
            keys=keys,
            targets=targets,
            cost=cost,
            strategy=strategy,
            cost_components=stored_components,
            candidate_rank=rank_hint,
        )
        signature = _candidate_signature(result)
        if signature in seen:
            return
        seen.add(signature)
        candidates.append(
            FingerAssignmentCandidate(result=result, base_cost=float(result.total_cost), rank=rank_hint)
        )

    add_candidate(base_cost, 0)
    base_result = candidates[0].result
    if top_k <= 1 or base_result.count == 0:
        return candidates

    perturb_scale = float(getattr(config, "assignment_top_k_extra_penalty", 1e-4))
    variant_scale = max(perturb_scale, 1e-3)
    perturb_targets = list(zip(base_result.assigned_finger_indices.tolist(), base_result.assigned_key_positions.tolist()))
    candidate_rank = 1
    for finger_index, key_position in perturb_targets:
        perturbed = np.array(base_cost, copy=True)
        perturbed[int(finger_index), int(key_position)] += perturb_scale * float(candidate_rank)
        add_candidate(perturbed, candidate_rank)
        candidate_rank += 1

    for first_index, (first_finger, first_key_position) in enumerate(perturb_targets):
        for second_finger, second_key_position in perturb_targets[first_index + 1 :]:
            if len(candidates) >= max(top_k * 3, top_k + 4):
                break
            perturbed = np.array(base_cost, copy=True)
            perturbed[int(first_finger), int(first_key_position)] += perturb_scale * float(candidate_rank)
            perturbed[int(second_finger), int(second_key_position)] += perturb_scale * float(candidate_rank)
            add_candidate(perturbed, candidate_rank)
            candidate_rank += 1

    middle_key = int(getattr(config, "assignment_middle_key", 44))
    hand_split = np.array(base_cost, copy=True)
    for finger_index in range(NUM_FINGERS):
        wrong_side = keys >= middle_key if is_left_finger(finger_index) else keys < middle_key
        hand_split[finger_index, wrong_side] += float(getattr(config, "assignment_wrong_hand_penalty", 1.0)) + variant_scale
    add_candidate(hand_split, candidate_rank)
    candidate_rank += 1

    if dense_previous is not None:
        hold_variant = np.array(base_cost, copy=True)
        for key_position, key in enumerate(keys.astype(np.int32).tolist()):
            owners = np.where(dense_previous == int(key))[0]
            if owners.size:
                hold_variant[:, key_position] += variant_scale
                hold_variant[int(owners[0]), key_position] -= 2.0 * variant_scale
        add_candidate(hold_variant, candidate_rank)
        candidate_rank += 1

    crossed_pairs = _crossed_selected_pairs(base_result)
    if crossed_pairs:
        crossing_variant = np.array(base_cost, copy=True)
        for finger_index, key_position in crossed_pairs:
            crossing_variant[int(finger_index), int(key_position)] += variant_scale
        add_candidate(crossing_variant, candidate_rank)

    return _rank_candidate_results(candidates, top_k)


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


def _sequence_unassigned_penalty(result: FingerAssignmentResult, config: object | None) -> float:
    if result.unassigned_keys.size == 0:
        return 0.0
    if bool(getattr(config, "assignment_fail_if_unassigned", False)):
        return float("inf")
    return float(getattr(config, "assignment_unassigned_penalty", 25.0)) * float(result.unassigned_keys.size)


def _assignment_future_heuristic(
    *,
    active_key_sequence: Sequence[np.ndarray],
    key_target_sequence: Sequence[np.ndarray],
    start_index: int,
    previous_fingertips: np.ndarray,
    previous_assignment: np.ndarray,
    config: object | None,
) -> float:
    horizon = max(int(getattr(config, "assignment_future_horizon", 0)), 0)
    if horizon <= 0:
        return 0.0
    discount = float(getattr(config, "assignment_sequence_cost_discount", 0.9))
    discount = min(max(discount, 0.0), 1.0)
    estimate = 0.0
    weight = discount
    fingertips = np.asarray(previous_fingertips, dtype=np.float32).copy()
    dense = np.asarray(previous_assignment, dtype=np.int32).copy()
    stop = min(len(active_key_sequence), start_index + 1 + horizon)
    for future_index in range(start_index + 1, stop):
        future_candidates = generate_assignment_candidates(
            active_key_sequence[future_index],
            fingertips,
            key_target_sequence[future_index],
            config,
            max_candidates=1,
            previous_assignment=dense,
        )
        if not future_candidates:
            continue
        candidate = future_candidates[0]
        crossing = assignment_crossing_penalty(
            candidate.result.assigned_finger_indices,
            candidate.result.assigned_keys,
            config,
        )
        local = (
            float(candidate.base_cost)
            + float(getattr(config, "assignment_crossing_weight", 0.0)) * float(crossing)
            + _sequence_unassigned_penalty(candidate.result, config)
        )
        estimate += weight * local
        fingertips = _assignment_next_fingertips(fingertips, candidate.result)
        dense = _assignment_next_keys(dense, candidate.result)
        weight *= discount
    return float(estimate)


def assign_fingers_sequence_beam(
    active_key_sequence: Sequence[np.ndarray],
    initial_fingertips: np.ndarray,
    key_target_sequence: Sequence[np.ndarray],
    *,
    config: object | None = None,
    beam_width: int | None = None,
    candidates_per_step: int | None = None,
) -> list[FingerAssignmentResult]:
    if len(active_key_sequence) != len(key_target_sequence):
        raise ValueError("active_key_sequence and key_target_sequence must have the same length")
    width = max(int(beam_width if beam_width is not None else getattr(config, "assignment_beam_width", 4)), 1)
    per_step = int(
        candidates_per_step
        if candidates_per_step is not None
        else getattr(config, "assignment_candidates_per_step", 0)
    )
    if per_step <= 0:
        per_step = int(getattr(config, "assignment_top_k", 1))
    per_step = max(per_step, 1)

    initial = _AssignmentBeamState(
        previous_fingertips=np.asarray(initial_fingertips, dtype=np.float32).copy(),
        previous_assignment=np.full((NUM_FINGERS,), -1, dtype=np.int32),
        cumulative_cost=0.0,
        results=(),
    )
    beam = [initial]
    for waypoint_index, (active_keys, key_targets) in enumerate(zip(active_key_sequence, key_target_sequence)):
        expansions: list[tuple[float, _AssignmentBeamState]] = []
        for state in beam:
            candidates = generate_assignment_candidates(
                active_keys,
                state.previous_fingertips,
                key_targets,
                config,
                max_candidates=per_step,
                previous_assignment=state.previous_assignment,
            )
            for candidate in candidates:
                crossing = assignment_crossing_penalty(
                    candidate.result.assigned_finger_indices,
                    candidate.result.assigned_keys,
                    config,
                )
                local_cost = (
                    float(candidate.base_cost)
                    + float(getattr(config, "assignment_crossing_weight", 0.0)) * float(crossing)
                    + _sequence_unassigned_penalty(candidate.result, config)
                )
                if not np.isfinite(local_cost):
                    continue
                next_fingertips = _assignment_next_fingertips(state.previous_fingertips, candidate.result)
                next_assignment = _assignment_next_keys(state.previous_assignment, candidate.result)
                cumulative = float(state.cumulative_cost + local_cost)
                heuristic = _assignment_future_heuristic(
                    active_key_sequence=active_key_sequence,
                    key_target_sequence=key_target_sequence,
                    start_index=waypoint_index,
                    previous_fingertips=next_fingertips,
                    previous_assignment=next_assignment,
                    config=config,
                )
                scored_result = replace(
                    candidate.result,
                    strategy="sequence_beam",
                    candidate_rank=int(candidate.rank),
                    candidate_score=float(cumulative + heuristic),
                )
                expansions.append(
                    (
                        float(cumulative + heuristic),
                        _AssignmentBeamState(
                            previous_fingertips=next_fingertips,
                            previous_assignment=next_assignment,
                            cumulative_cost=cumulative,
                            results=state.results + (scored_result,),
                        ),
                    )
                )
        if not expansions:
            raise RuntimeError("sequence_beam found no feasible assignment path")
        expansions.sort(
            key=lambda item: (
                float(item[0]),
                tuple(item[1].previous_assignment.astype(int).tolist()),
            )
        )
        beam = [state for _, state in expansions[:width]]
    beam.sort(key=lambda state: (float(state.cumulative_cost), tuple(state.previous_assignment.astype(int).tolist())))
    return list(beam[0].results) if beam else []


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

    if strategy in {"composite_cost", "ik_aware_topk", "sequence_beam"}:
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
