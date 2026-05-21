from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bagatelle.assignment import (  # noqa: E402
    build_composite_assignment_cost,
    assign_fingers_previous_pose,
    generate_assignment_candidates,
)
from bagatelle.config import BagatelleConfig  # noqa: E402


def test_legacy_mode_matches_default_none() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    fingertips[0] = [0.0, 0.0, 0.0]
    fingertips[1] = [1.0, 0.0, 0.0]
    keys = np.asarray([10, 11], dtype=np.int32)
    targets = np.asarray([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32)

    legacy_none = assign_fingers_previous_pose(keys, fingertips, targets, None)
    legacy_cfg = assign_fingers_previous_pose(keys, fingertips, targets, BagatelleConfig())

    np.testing.assert_array_equal(legacy_none.assigned_finger_indices, legacy_cfg.assigned_finger_indices)
    np.testing.assert_array_equal(legacy_none.assigned_keys, legacy_cfg.assigned_keys)
    np.testing.assert_allclose(legacy_none.cost_matrix, legacy_cfg.cost_matrix)


def test_composite_mode_assigns_unique_fingers_and_keys() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    fingertips[0] = [0.0, 0.0, 0.0]
    fingertips[1] = [1.0, 0.0, 0.0]
    fingertips[5] = [2.0, 0.0, 0.0]
    cfg = BagatelleConfig(assignment_strategy="composite_cost")
    result = assign_fingers_previous_pose(
        np.asarray([20, 30, 40], dtype=np.int32),
        fingertips,
        np.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]], dtype=np.float32),
        cfg,
    )

    assert len(set(result.assigned_finger_indices.tolist())) == result.count
    assert len(set(result.assigned_keys.tolist())) == result.count


def test_wrong_hand_penalty_changes_cost_matrix() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    targets = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    cfg = BagatelleConfig(
        assignment_strategy="composite_cost",
        assignment_hand_zone_weight=1.0,
        assignment_wrong_hand_penalty=3.0,
        assignment_middle_key=44,
    )
    cost, _ = build_composite_assignment_cost(
        np.asarray([20, 60], dtype=np.int32),
        targets,
        fingertips,
        cfg,
    )

    assert cost[5, 0] > cost[0, 0]
    assert cost[0, 1] > cost[5, 1]


def test_held_note_persistence_prefers_previous_finger() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    cfg = BagatelleConfig(
        assignment_strategy="composite_cost",
        assignment_hold_weight=2.0,
    )
    cost, _ = build_composite_assignment_cost(
        np.asarray([40], dtype=np.int32),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        fingertips,
        cfg,
        previous_assignment=np.asarray([-1, -1, 40, -1, -1, -1, -1, -1, -1, -1], dtype=np.int32),
    )

    assert cost[2, 0] < cost[1, 0]


def test_candidate_generation_is_deterministic() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    fingertips[0] = [0.0, 0.0, 0.0]
    fingertips[1] = [0.8, 0.0, 0.0]
    cfg = BagatelleConfig(
        assignment_strategy="composite_cost",
        assignment_top_k=3,
        assignment_top_k_extra_penalty=1e-3,
    )

    first = generate_assignment_candidates(
        np.asarray([10, 11], dtype=np.int32),
        fingertips,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        cfg,
    )
    second = generate_assignment_candidates(
        np.asarray([10, 11], dtype=np.int32),
        fingertips,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        cfg,
    )

    assert [
        (candidate.result.assigned_finger_indices.tolist(), candidate.result.assigned_keys.tolist())
        for candidate in first
    ] == [
        (candidate.result.assigned_finger_indices.tolist(), candidate.result.assigned_keys.tolist())
        for candidate in second
    ]


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError):
        assign_fingers_previous_pose(
            np.asarray([10], dtype=np.int32),
            np.zeros((10, 3), dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            BagatelleConfig(assignment_strategy="bad"),
        )
