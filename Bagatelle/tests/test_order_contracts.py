from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bagatelle.assignment import LEFT_HAND_FINGERS, RIGHT_HAND_FINGERS  # noqa: E402
from bagatelle.kinematics import FINGER_ORDER, HAND_STATE_DIM, JOINT_INDEX_RANGES_BY_HAND, JOINT_ORDER  # noqa: E402


def test_assignment_finger_order_contract_is_left_then_right() -> None:
    assert FINGER_ORDER == "left_hand_sites_then_right_hand_sites"
    assert LEFT_HAND_FINGERS == (0, 1, 2, 3, 4)
    assert RIGHT_HAND_FINGERS == (5, 6, 7, 8, 9)


def test_reduced_joint_order_contract_is_right_then_left() -> None:
    assert JOINT_ORDER == "right_hand_joints_then_left_hand_joints"
    assert HAND_STATE_DIM == 46
    assert JOINT_INDEX_RANGES_BY_HAND == {"right_hand": [0, 23], "left_hand": [23, 46]}
