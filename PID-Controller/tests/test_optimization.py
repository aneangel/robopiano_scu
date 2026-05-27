from __future__ import annotations

from pid_controller.optimization import make_gain_candidates, metric_from_row, score_pid_result


def test_make_gain_candidates_removes_unused_terms_by_controller_kind() -> None:
    candidates = make_gain_candidates(
        controllers=("p", "pd", "pid"),
        kp_values=(1.0,),
        kd_values=(0.01, 0.02),
        ki_values=(0.03,),
        integral_limits=(0.1,),
        setpoint_policies=("minimum_jerk",),
        target_velocity_scales=(0.0, 1.0),
        sustain_values=(0.0,),
    )

    p = [candidate for candidate in candidates if candidate.controller_kind == "p"]
    pd = [candidate for candidate in candidates if candidate.controller_kind == "pd"]
    pid = [candidate for candidate in candidates if candidate.controller_kind == "pid"]

    assert {candidate.kd for candidate in p} == {0.0}
    assert {candidate.ki for candidate in p} == {0.0}
    assert {candidate.ki for candidate in pd} == {0.0}
    assert {candidate.ki for candidate in pid} == {0.03}
    assert len(pd) == 4


def test_score_pid_result_rewards_f1_and_penalizes_tracking_error() -> None:
    good = {
        "event_f1": 0.8,
        "frame_f1": 0.7,
        "rp1m_key_f1": 0.6,
        "hand_qpos_l2_vs_reference": {"mean": 0.5},
        "terminated": False,
    }
    bad = {
        "event_f1": 0.7,
        "frame_f1": 0.7,
        "rp1m_key_f1": 0.6,
        "hand_qpos_l2_vs_reference": {"mean": 4.0},
        "terminated": True,
    }

    assert score_pid_result(good) > score_pid_result(bad)


def test_metric_from_row_flips_hand_l2_for_sorting() -> None:
    assert metric_from_row({"mean_hand_l2": 0.25}, "hand_l2_mean") == -0.25
    assert metric_from_row({"mean_hand_l2_final": 0.35}, "hand_l2_final") == -0.35
    assert metric_from_row({"mean_hand_l2_last_third_mean": 0.45}, "hand_l2_last_third_mean") == -0.45
    assert metric_from_row({"mean_one_to_one_l2_mean": 0.2}, "one_to_one_l2_mean") == -0.2
    assert metric_from_row({"mean_one_to_one_l2_final": 0.3}, "one_to_one_l2_final") == -0.3
    assert metric_from_row({"mean_coupled_l2_mean": 0.3}, "coupled_l2_mean") == -0.3
    assert metric_from_row({"mean_actuator_l2_mean": 0.4}, "actuator_l2_mean") == -0.4
    assert metric_from_row({"mean_event_f1": 0.5}, "event_f1") == 0.5
