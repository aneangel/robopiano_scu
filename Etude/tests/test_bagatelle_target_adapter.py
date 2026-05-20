from __future__ import annotations

import numpy as np
import pytest

from etude.data.bagatelle_targets import (
    assignment_masks,
    dense_fingertip_targets_from_waypoints,
    normalize_bagatelle_metadata,
)


def test_dense_fingertip_targets_from_waypoints_expands_with_hold_fill() -> None:
    targets = np.stack(
        [
            np.full((10, 3), 1.0, dtype=np.float32),
            np.full((10, 3), 2.0, dtype=np.float32),
        ],
        axis=0,
    )
    dense = dense_fingertip_targets_from_waypoints(targets, np.array([0, 2]), total_steps=4)
    assert dense.shape == (4, 10, 3)
    np.testing.assert_allclose(dense[0], 1.0)
    np.testing.assert_allclose(dense[1], 1.0)
    np.testing.assert_allclose(dense[2], 2.0)
    np.testing.assert_allclose(dense[3], 2.0)


def test_assignment_masks_create_active_and_inactive_fingers() -> None:
    active, inactive = assignment_masks(np.array([[0, -1, 5, -1, -1, 3, -1, -1, -1, 2]], dtype=np.int32))
    np.testing.assert_array_equal(active[0], np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(inactive[0], 1.0 - active[0])


def test_assignment_masks_follow_target_key_activity_windows() -> None:
    assignments = np.full((4, 10), -1, dtype=np.int32)
    assignments[:, 0] = 7
    target_keys = np.zeros((4, 88), dtype=np.float32)
    target_keys[1:3, 7] = 1.0
    active, inactive = assignment_masks(assignments, target_keys=target_keys, total_steps=4)
    np.testing.assert_array_equal(active[:, 0], np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(inactive[:, 0], np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_normalize_bagatelle_metadata_adds_dense_fields() -> None:
    metadata = {
        "waypoint_frames": np.array([0, 2], dtype=np.int32),
        "fingertip_targets": np.stack(
            [
                np.full((10, 3), 0.25, dtype=np.float32),
                np.full((10, 3), 0.75, dtype=np.float32),
            ],
            axis=0,
        ),
        "assignments": np.array([[0] * 10, [-1] * 10], dtype=np.int32),
        "assignment_costs": np.array([[0.5] * 10, [np.nan] * 10], dtype=np.float32),
        "target_keys": np.zeros((4, 88), dtype=np.float32),
    }
    normalized = normalize_bagatelle_metadata(metadata, q_ref_len=4, strict=True)
    assert normalized["desired_fingertips"].shape == (4, 10, 3)
    assert normalized["fingertip_weights"].shape == (4, 10)
    assert normalized["active_finger_mask"].shape == (4, 10)
    assert normalized["inactive_finger_mask"].shape == (4, 10)
    np.testing.assert_allclose(normalized["desired_fingertips"][1], 0.25)
    np.testing.assert_allclose(normalized["desired_fingertips"][3], 0.75)
    assert np.all(np.isfinite(normalized["fingertip_weights"]))


def test_normalize_bagatelle_metadata_leaves_legacy_metadata_unchanged() -> None:
    metadata = {"source": "unit"}
    normalized = normalize_bagatelle_metadata(metadata, q_ref_len=3, strict=False)
    assert normalized == metadata


def test_normalize_bagatelle_metadata_strict_mode_rejects_bad_shapes() -> None:
    metadata = {
        "fingertip_targets": np.zeros((2, 9, 3), dtype=np.float32),
        "waypoint_frames": np.array([0, 1], dtype=np.int32),
    }
    with pytest.raises(ValueError):
        normalize_bagatelle_metadata(metadata, q_ref_len=2, strict=True)
