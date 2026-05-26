from __future__ import annotations

import numpy as np

from ballade.action_masks import full_mask, left_hand_mask, mask_for_goal_keys, right_hand_mask, sustain_mask


def test_39d_masks_follow_reduced_action_convention() -> None:
    assert full_mask(39).sum() == 39
    right = right_hand_mask(39)
    left = left_hand_mask(39)
    sustain = sustain_mask(39)
    assert right[:19].all()
    assert not right[19:].any()
    assert left[19:38].all()
    assert not left[:19].any()
    assert sustain[38]
    assert sustain.sum() == 1


def test_mask_for_goal_keys_uses_assignment() -> None:
    goal = np.zeros((88,), dtype=np.float32)
    goal[60] = 1.0
    mask = mask_for_goal_keys(goal, {60: "right"}, action_dim=39)
    assert mask[:19].all()
    assert not mask[19:38].any()
    assert not mask[38]


def test_non_39d_masks_do_not_hardcode_widths() -> None:
    assert right_hand_mask(10).sum() == 5
    assert left_hand_mask(10).sum() == 5
    assert sustain_mask(10)[-1]
