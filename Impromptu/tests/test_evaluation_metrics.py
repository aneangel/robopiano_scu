from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from impromptu.evaluation import evaluate_trajectory_payload  # noqa: E402


def _fake_payload() -> dict[str, np.ndarray]:
    target_keys = np.zeros((5, 88), dtype=np.float32)
    target_keys[1, 10] = 1.0
    target_keys[3, 20] = 1.0

    waypoint_frames = np.asarray([1, 3], dtype=np.int64)
    waypoint_target_keys = np.zeros((2, 88), dtype=np.float32)
    waypoint_target_keys[0, [10, 11]] = 1.0
    waypoint_target_keys[1, 20] = 1.0

    assignments = np.full((2, 10), -1, dtype=np.int32)
    assignments[0, 0] = 10
    assignments[0, 1] = 11
    assignments[1, 2] = 20

    unassigned_keys = np.full((2, 10), -1, dtype=np.int32)
    unassigned_keys[1, 0] = 33

    fingertip_targets = np.full((2, 10, 3), np.nan, dtype=np.float32)
    fingertip_targets[0, 0] = np.asarray([0.10, 0.00, 0.00], dtype=np.float32)
    fingertip_targets[0, 1] = np.asarray([0.20, 0.00, 0.00], dtype=np.float32)
    fingertip_targets[1, 2] = np.asarray([0.30, 0.00, 0.00], dtype=np.float32)

    waypoint_fingertips = np.full((2, 10, 3), np.nan, dtype=np.float32)
    waypoint_fingertips[0, 0] = np.asarray([0.11, 0.00, 0.00], dtype=np.float32)
    waypoint_fingertips[0, 1] = np.asarray([0.22, 0.00, 0.00], dtype=np.float32)
    waypoint_fingertips[1, 2] = np.asarray([0.29, 0.00, 0.00], dtype=np.float32)

    dense_targets = np.full((20, 10, 3), np.nan, dtype=np.float32)
    dense_weights = np.zeros((20, 10), dtype=np.float32)
    dense_targets[4, 0] = np.asarray([0.10, 0.00, 0.00], dtype=np.float32)
    dense_targets[4, 1] = np.asarray([0.20, 0.00, 0.00], dtype=np.float32)
    dense_targets[8, 2] = np.asarray([0.25, 0.00, 0.00], dtype=np.float32)
    dense_targets[12, 2] = np.asarray([0.30, 0.00, 0.00], dtype=np.float32)
    dense_targets[19, 2] = np.asarray([0.35, 0.00, 0.00], dtype=np.float32)
    dense_weights[4, [0, 1]] = 1.0
    dense_weights[8, 2] = 0.5
    dense_weights[12, 2] = 1.0
    dense_weights[19, 2] = 0.25

    ik_anchor_frames_dense = np.asarray([0, 4, 8, 12, 19], dtype=np.int64)
    ik_anchor_frames_control = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    ik_anchor_fingertips = np.zeros((5, 10, 3), dtype=np.float32)
    ik_anchor_fingertips[1, 0] = np.asarray([0.105, 0.00, 0.00], dtype=np.float32)
    ik_anchor_fingertips[1, 1] = np.asarray([0.205, 0.00, 0.00], dtype=np.float32)
    ik_anchor_fingertips[2, 2] = np.asarray([0.255, 0.00, 0.00], dtype=np.float32)
    ik_anchor_fingertips[3, 2] = np.asarray([0.305, 0.00, 0.00], dtype=np.float32)
    ik_anchor_fingertips[4, 2] = np.asarray([0.345, 0.00, 0.00], dtype=np.float32)
    ik_anchor_metrics = np.asarray(
        [
            [1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    planned_hand_joints = np.linspace(0.0, 1.0, num=5 * 46, dtype=np.float32).reshape(5, 46)
    planned_hand_velocities = np.linspace(0.0, 0.4, num=5 * 46, dtype=np.float32).reshape(5, 46)

    return {
        "target_keys": target_keys,
        "waypoint_frames": waypoint_frames,
        "waypoint_target_keys": waypoint_target_keys,
        "assignments": assignments,
        "unassigned_keys": unassigned_keys,
        "fingertip_targets": fingertip_targets,
        "waypoint_fingertips": waypoint_fingertips,
        "fingertip_trajectory_targets": dense_targets,
        "fingertip_trajectory_weights": dense_weights,
        "ik_anchor_frames_dense": ik_anchor_frames_dense,
        "ik_anchor_frames_control": ik_anchor_frames_control,
        "ik_anchor_fingertips": ik_anchor_fingertips,
        "ik_anchor_metrics": ik_anchor_metrics,
        "planned_hand_joints": planned_hand_joints,
        "planned_hand_velocities": planned_hand_velocities,
    }


def test_evaluate_trajectory_payload_reports_planner_native_metrics() -> None:
    metrics = evaluate_trajectory_payload(_fake_payload())

    assert metrics["assignment_rate"] == 1.0
    assert metrics["num_waypoints"] == 2
    assert metrics["num_target_key_events"] == 3
    assert metrics["num_assigned_key_events"] == 3
    assert metrics["num_unassigned_key_events"] == 1
    assert metrics["waypoint_fingertip_error_p95"] >= 0.0
    assert metrics["exact_waypoint_sparse_error_p95"] >= 0.0
    assert metrics["exact_waypoint_anchor_error_p95"] >= 0.0
    assert 0.0 <= metrics["exact_waypoint_anchor_success_rate_020m"] <= 1.0
    assert metrics["ik_anchor_error_weight_ge_1_p95"] >= 0.0
    assert 0.0 <= metrics["ik_anchor_success_rate_020m_weight_ge_1"] <= 1.0
    assert metrics["press_frame_anchor_error_p95"] >= 0.0
    assert metrics["prepress_approach_anchor_error_p95"] >= 0.0
    assert metrics["hold_end_anchor_error_p95"] >= 0.0
    assert metrics["release_anchor_error_p95"] >= 0.0
    assert metrics["inactive_clearance_violation_count"] >= 0
    assert metrics["wrong_key_proximity_violation_count"] >= 0
    assert metrics["inactive_clearance_violation_rate_per_checked_anchor"] >= 0.0
    assert metrics["wrong_key_proximity_violation_rate_per_checked_anchor"] >= 0.0
    assert metrics["ik_anchor_failure_count"] >= 0
    assert metrics["ik_anchor_optimizer_failure_count"] >= 0
    assert metrics["exact_waypoint_anchor_error_finger_0_p95"] >= 0.0
    assert metrics["exact_waypoint_anchor_error_polyphony_2_p95"] >= 0.0
    assert 0.0 <= metrics["waypoint_success_rate_010m"] <= 1.0
    assert 0.0 <= metrics["waypoint_dense_activity_rate"] <= 1.0
    assert 0.0 <= metrics["waypoint_has_exact_anchor_rate"] <= 1.0
    assert metrics["online_metrics_status"] == "absent"
    assert metrics["online_metrics_available"] is False


def test_evaluate_trajectory_payload_empty_payload_does_not_crash() -> None:
    metrics = evaluate_trajectory_payload({})

    assert metrics["num_waypoints"] == 0
    assert metrics["assignment_rate"] == 0.0
    assert metrics["dense_num_frames"] == 0
    assert metrics["waypoint_dense_inactive_count"] == 0
    assert metrics["waypoint_missing_exact_anchor_count"] == 0
    assert metrics["exact_waypoint_sparse_error_p95"] == 0.0
    assert metrics["exact_waypoint_anchor_error_p95"] == 0.0
    assert metrics["ik_anchor_error_weight_ge_1_p95"] == 0.0
    assert metrics["press_frame_anchor_error_p95"] == 0.0
    assert metrics["online_metrics_status"] == "absent"
    assert metrics["online_metrics_available"] is False
