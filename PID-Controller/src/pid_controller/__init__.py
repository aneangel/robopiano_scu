"""PID hand-state tracking controller for RoboPianist/RP1M rollouts."""

from pid_controller.action_space import (
    FULL_ENV_ACTION_DIM,
    FULL_HAND_ACTION_DIM,
    FULL_HAND_STATE_DIM,
    REDUCED_ENV_ACTION_DIM,
    REDUCED_HAND_ACTION_DIM,
    REDUCED_HAND_STATE_DIM,
    ActionSpaceSummary,
    expected_action_space_summaries,
    expected_action_space_summary,
)
from pid_controller.controller import (
    HandPIDController,
    PIDControllerConfig,
    PIDGains,
    make_controller_config,
)
from pid_controller.mapping import (
    REDUCED_ACTION_DIM,
    REDUCED_ACTION_DIM_PER_HAND,
    ActionJointMapEntry,
    action_signals_from_hand_state,
    build_reduced_action_mapping,
    joint_index_groups,
)
from pid_controller.optimization import GainCandidate, make_gain_candidates, score_pid_result
from pid_controller.rollout import run_impromptu_pid_rollout

__all__ = [
    "ActionJointMapEntry",
    "ActionSpaceSummary",
    "FULL_ENV_ACTION_DIM",
    "FULL_HAND_ACTION_DIM",
    "FULL_HAND_STATE_DIM",
    "HandPIDController",
    "PIDControllerConfig",
    "PIDGains",
    "REDUCED_ENV_ACTION_DIM",
    "REDUCED_HAND_ACTION_DIM",
    "REDUCED_HAND_STATE_DIM",
    "REDUCED_ACTION_DIM",
    "REDUCED_ACTION_DIM_PER_HAND",
    "action_signals_from_hand_state",
    "build_reduced_action_mapping",
    "joint_index_groups",
    "expected_action_space_summaries",
    "expected_action_space_summary",
    "GainCandidate",
    "make_controller_config",
    "make_gain_candidates",
    "run_impromptu_pid_rollout",
    "score_pid_result",
]
