from __future__ import annotations

import numpy as np

from allegro.hand_error import compute_source_hand_errors, summarize_error_prefixes


def test_hand_errors_compare_after_step_to_next_target() -> None:
    sim = np.array([[1.0, 1.0], [3.0, 5.0]], dtype=np.float32)
    target = np.array([[0.0, 0.0], [1.0, 2.0], [5.0, 5.0]], dtype=np.float32)
    errors = compute_source_hand_errors(sim_hand_joints=sim, target_hand_joints=target, dt=0.05)

    assert errors["source_step"].tolist() == [0, 1]
    assert errors["target_step"].tolist() == [1, 2]
    np.testing.assert_allclose(errors["hand_l2"], [1.0, 2.0])
    np.testing.assert_allclose(errors["elapsed_s"], [0.05, 0.10])


def test_prefix_summary_reports_short_horizons() -> None:
    sim = np.zeros((5, 2), dtype=np.float32)
    target = np.zeros((6, 2), dtype=np.float32)
    target[1:, 0] = np.arange(5, dtype=np.float32)
    errors = compute_source_hand_errors(sim_hand_joints=sim, target_hand_joints=target)
    rows = summarize_error_prefixes(errors, prefixes=[2, 4])

    assert [row["source_steps"] for row in rows] == [2, 4, 5]
    assert rows[0]["hand_l2_final"] == 1.0
    assert rows[-1]["hand_l2_final"] == 4.0
