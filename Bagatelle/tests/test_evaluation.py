from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bagatelle.evaluation import (  # noqa: E402
    _action_summary,
    _build_reference_schedule,
    metric_validity_for_rollout,
    fingertip_summary_from_trajectory,
)


def test_fingertip_summary_ignores_unassigned_nan_targets() -> None:
    targets = np.full((1, 10, 3), np.nan, dtype=np.float32)
    measured = np.zeros((1, 10, 3), dtype=np.float32)
    targets[0, 2] = [0.0, 0.0, 0.0]
    measured[0, 2] = [0.0, 0.01, 0.0]

    summary = fingertip_summary_from_trajectory(
        {"fingertip_targets": targets, "waypoint_fingertips": measured},
        success_threshold=0.02,
    )

    assert summary["fingertip_assignments"] == 1
    assert summary["fingertip_distance_mean"] == np.float32(0.01)
    assert summary["fingertip_width_distance_mean"] == np.float32(0.01)
    assert summary["fingertip_width_success_rate"] == 1.0
    assert summary["fingertip_success_rate"] == 1.0


class _ActionSpec:
    shape = (3,)
    dtype = np.float32
    minimum = -np.ones(3, dtype=np.float32)
    maximum = np.ones(3, dtype=np.float32)


def test_metric_validity_rejects_direct_pose_replay_as_actuated() -> None:
    target = np.zeros((2, 88), dtype=np.float32)
    target[0, 10] = 1.0
    played = target.copy()

    validity = metric_validity_for_rollout(
        {
            "control_mode": "diagnostic_pose_replay_direct_hand_qpos_pose_injection_with_settle",
            "is_actuated_playback": False,
            "action_dim": None,
            "mapped_actuators": 0,
            "actions_executed": 0,
            "pose_frames_applied": 2,
        },
        target_keys=target,
        played_keys=played,
    )

    assert validity["is_actuated_playback"] is False
    assert validity["actions_executed_positive"] is False
    assert validity["mapped_actuators_or_action_dim_positive"] is False
    assert validity["no_direct_pose_injection_in_scoring_loop"] is False
    assert validity["target_not_copied_to_played_roll"] is False


def test_metric_validity_does_not_flag_matching_silence_as_target_copy() -> None:
    silence = np.zeros((2, 88), dtype=np.float32)

    validity = metric_validity_for_rollout(
        {
            "control_mode": "actuated_env_step_pd",
            "is_actuated_playback": True,
            "action_dim": 3,
            "mapped_actuators": 3,
            "actions_executed": 2,
            "pose_frames_applied": 0,
            "controller_input_uses_colorization": False,
        },
        target_keys=silence,
        played_keys=silence.copy(),
    )

    assert validity["target_not_copied_to_played_roll"] is True


def test_metric_validity_accepts_env_step_rollout_with_actions() -> None:
    target = np.zeros((2, 88), dtype=np.float32)
    target[0, 10] = 1.0
    played = np.zeros((2, 88), dtype=np.float32)

    validity = metric_validity_for_rollout(
        {
            "control_mode": "actuated_env_step_pd",
            "is_actuated_playback": True,
            "action_dim": 3,
            "mapped_actuators": 3,
            "actions_executed": 2,
            "pose_frames_applied": 0,
            "controller_input_uses_colorization": False,
        },
        target_keys=target,
        played_keys=played,
    )

    assert validity["is_actuated_playback"] is True
    assert validity["actions_executed_positive"] is True
    assert validity["mapped_actuators_or_action_dim_positive"] is True
    assert validity["no_direct_pose_injection_in_scoring_loop"] is True
    assert validity["target_not_copied_to_played_roll"] is True
    assert validity["no_colorization_signal_used_as_control_input"] is True


def test_action_summary_counts_bound_hits() -> None:
    summary = _action_summary(
        np.asarray([[0.0, 1.0, 0.2], [0.5, 0.25, -1.0]], dtype=np.float32),
        _ActionSpec(),
    )

    assert summary["actions_executed"] == 2
    assert summary["action_abs_max"] == 1.0
    assert summary["action_bound_hit_rate"] == 1.0


def test_reference_schedule_repeats_for_faster_env_dt() -> None:
    schedule = _build_reference_schedule(horizon=3, trajectory_dt=0.02, env_dt=0.01, max_steps=None)

    assert schedule.tolist() == [0, 0, 1, 1, 2, 2]
