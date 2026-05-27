from __future__ import annotations

import numpy as np

from pid_controller.mapping import (
    REDUCED_ACTION_DIM,
    action_signals_from_hand_state,
    build_reduced_action_mapping,
    joint_index_groups,
)


def test_nominal_mapping_covers_reduced_actions() -> None:
    mapping = build_reduced_action_mapping()

    assert len(mapping) == REDUCED_ACTION_DIM
    assert [entry.action_index for entry in mapping] == list(range(REDUCED_ACTION_DIM))
    assert mapping[0].joint_indices == (0,)
    assert mapping[1].joint_indices == (1,)
    assert mapping[7].kind == "fixed_tendon"
    assert mapping[7].joint_indices == (4, 5)
    assert mapping[17].joint_indices == (21,)
    assert mapping[18].joint_indices == (22,)
    assert mapping[19].joint_indices == (23,)
    assert mapping[36].joint_indices == (44,)
    assert mapping[37].joint_indices == (45,)
    assert mapping[38].kind == "sustain"


def test_action_projection_uses_weighted_least_squares_for_tendon_joints() -> None:
    mapping = build_reduced_action_mapping()
    qpos = np.zeros((46,), dtype=np.float32)
    qpos[4] = 0.25
    qpos[5] = 0.75
    qpos[21] = 0.03
    qpos[45] = 0.04

    action = action_signals_from_hand_state(qpos, mapping)

    assert action[7] == np.float32(0.5)
    assert action[17] == np.float32(0.03)
    assert action[37] == np.float32(0.04)
    assert action[38] == np.float32(0.0)


def test_action_projection_keeps_legacy_sum_available() -> None:
    mapping = build_reduced_action_mapping()
    qpos = np.zeros((46,), dtype=np.float32)
    qpos[4] = 0.25
    qpos[5] = 0.75

    action = action_signals_from_hand_state(qpos, mapping, projection="legacy_sum")

    assert action[7] == np.float32(1.0)


def test_joint_groups_separate_one_to_one_from_coupled() -> None:
    groups = joint_index_groups(build_reduced_action_mapping())

    assert 0 in groups["one_to_one"]
    assert 4 not in groups["one_to_one"]
    assert 4 in groups["coupled"]
    assert 5 in groups["coupled"]
    assert len(groups["one_to_one"]) == 30
    assert len(groups["coupled"]) == 16


def test_mapping_accepts_live_style_names() -> None:
    hand_names = []
    nominal = [entry.joint_names[0] for entry in build_reduced_action_mapping()[:19] if entry.joint_names]
    # Explicit order mirrors Bagatelle/Impromptu: right 23 joints, then left 23.
    per_hand = (
        "WRJ2",
        "WRJ1",
        "FFJ4",
        "FFJ3",
        "FFJ2",
        "FFJ1",
        "MFJ4",
        "MFJ3",
        "MFJ2",
        "MFJ1",
        "RFJ4",
        "RFJ3",
        "RFJ2",
        "RFJ1",
        "LFJ4",
        "LFJ3",
        "LFJ2",
        "LFJ1",
        "THJ4",
        "THJ3",
        "THJ2",
        "forearm_tx",
        "forearm_ty",
    )
    hand_names.extend(f"right_hand/rh_{name}" for name in per_hand)
    hand_names.extend(f"left_hand/lh_{name}" for name in per_hand)
    actuator_names = []
    per_hand_act = (
        "A_WRJ2",
        "A_WRJ1",
        "A_THJ4",
        "A_THJ3",
        "A_THJ2",
        "A_FFJ4",
        "A_FFJ3",
        "A_FFJ0",
        "A_MFJ4",
        "A_MFJ3",
        "A_MFJ0",
        "A_RFJ4",
        "A_RFJ3",
        "A_RFJ0",
        "A_LFJ4",
        "A_LFJ3",
        "A_LFJ0",
        "forearm_tx",
        "forearm_ty",
    )
    actuator_names.extend(f"right_hand/rh_{name}" for name in per_hand_act)
    actuator_names.extend(f"left_hand/lh_{name}" for name in per_hand_act)
    actuator_names.append("sustain")

    mapping = build_reduced_action_mapping(hand_joint_names=hand_names, actuator_names=actuator_names)

    assert nominal
    assert mapping[7].joint_names == ("right_hand/rh_FFJ2", "right_hand/rh_FFJ1")
    assert mapping[36].joint_names == ("left_hand/lh_forearm_tx",)
