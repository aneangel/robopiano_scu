from __future__ import annotations

import numpy as np

from etude.evaluation.fingertip_metrics import compute_fingertip_assignment_metrics


def test_fingertip_metrics_ignore_inactive_fingers() -> None:
    current = np.zeros((1, 10, 3), dtype=np.float32)
    desired = np.zeros((1, 10, 3), dtype=np.float32)
    desired[0, 0, 0] = 0.01
    desired[0, 1, 0] = 1.0
    active = np.zeros((1, 10), dtype=np.float32)
    active[0, 0] = 1.0
    metrics = compute_fingertip_assignment_metrics(current, desired, active_finger_mask=active)
    assert np.isclose(metrics["fingertip/active_l2_mean"], 0.01)
    assert np.isclose(metrics["fingertip/assigned_success_rate_at_1cm"], 1.0)


def test_fingertip_metrics_compute_success_rates() -> None:
    current = np.zeros((3, 10, 3), dtype=np.float32)
    desired = np.zeros((3, 10, 3), dtype=np.float32)
    desired[0, 0, 0] = 0.005
    desired[1, 0, 0] = 0.03
    desired[2, 0, 0] = 0.06
    active = np.zeros((3, 10), dtype=np.float32)
    active[:, 0] = 1.0
    metrics = compute_fingertip_assignment_metrics(current, desired, active_finger_mask=active)
    assert np.isclose(metrics["fingertip/assigned_success_rate_at_1cm"], 1.0 / 3.0)
    assert np.isclose(metrics["fingertip/assigned_success_rate_at_2cm"], 1.0 / 3.0)
    assert np.isclose(metrics["fingertip/assigned_success_rate_at_5cm"], 2.0 / 3.0)


def test_fingertip_metrics_default_active_mask_uses_finite_targets() -> None:
    current = np.zeros((1, 10, 3), dtype=np.float32)
    desired = np.full((1, 10, 3), np.nan, dtype=np.float32)
    desired[0, 2] = np.array([0.02, 0.0, 0.0], dtype=np.float32)
    metrics = compute_fingertip_assignment_metrics(current, desired)
    assert np.isclose(metrics["fingertip/active_l2_mean"], 0.02)


def test_fingertip_metrics_return_empty_dict_for_empty_inputs() -> None:
    metrics = compute_fingertip_assignment_metrics(
        np.zeros((0, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.float32),
    )
    assert metrics == {}
