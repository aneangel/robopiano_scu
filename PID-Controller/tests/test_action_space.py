from __future__ import annotations

from pid_controller.action_space import (
    FULL_ENV_ACTION_DIM,
    FULL_HAND_ACTION_DIM,
    FULL_HAND_ACTION_DIM_PER_HAND,
    FULL_HAND_JOINT_DIM_PER_HAND,
    FULL_HAND_STATE_DIM,
    REDUCED_ENV_ACTION_DIM,
    REDUCED_HAND_ACTION_DIM,
    REDUCED_HAND_ACTION_DIM_PER_HAND,
    REDUCED_HAND_JOINT_DIM_PER_HAND,
    REDUCED_HAND_STATE_DIM,
    expected_action_space_summary,
)


def test_reduced_space_matches_impromptu_46d_hand_state_but_not_direct_actions() -> None:
    summary = expected_action_space_summary(reduced_action_space=True)

    assert summary.hand_action_dim_per_hand == REDUCED_HAND_ACTION_DIM_PER_HAND == 19
    assert summary.hand_joint_dim_per_hand == REDUCED_HAND_JOINT_DIM_PER_HAND == 23
    assert summary.hand_action_dim == REDUCED_HAND_ACTION_DIM == 38
    assert summary.hand_state_dim == REDUCED_HAND_STATE_DIM == 46
    assert summary.env_action_dim == REDUCED_ENV_ACTION_DIM == 39
    assert not summary.has_46_hand_actions_for_46_hand_states
    assert not summary.has_46_env_actions_for_46_hand_states
    assert not summary.hand_action_state_dims_match


def test_full_space_is_not_a_46_to_46_map() -> None:
    summary = expected_action_space_summary(reduced_action_space=False)

    assert summary.hand_action_dim_per_hand == FULL_HAND_ACTION_DIM_PER_HAND == 22
    assert summary.hand_joint_dim_per_hand == FULL_HAND_JOINT_DIM_PER_HAND == 26
    assert summary.hand_action_dim == FULL_HAND_ACTION_DIM == 44
    assert summary.hand_state_dim == FULL_HAND_STATE_DIM == 52
    assert summary.env_action_dim == FULL_ENV_ACTION_DIM == 45
    assert not summary.has_46_hand_actions_for_46_hand_states
    assert not summary.has_46_env_actions_for_46_hand_states
    assert not summary.hand_action_state_dims_match
