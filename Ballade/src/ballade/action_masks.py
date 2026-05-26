from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ballade.constants import LEFT_HAND_SLICE_39D, REDUCED_ACTION_DIM, RIGHT_HAND_SLICE_39D, SUSTAIN_INDEX_39D


def full_mask(action_dim: int) -> np.ndarray:
    return np.ones((int(action_dim),), dtype=bool)


def right_hand_mask(action_dim: int) -> np.ndarray:
    dim = int(action_dim)
    mask = np.zeros((dim,), dtype=bool)
    if dim == REDUCED_ACTION_DIM:
        mask[RIGHT_HAND_SLICE_39D] = True
    else:
        mask[: dim // 2] = True
    return mask


def left_hand_mask(action_dim: int) -> np.ndarray:
    dim = int(action_dim)
    mask = np.zeros((dim,), dtype=bool)
    if dim == REDUCED_ACTION_DIM:
        mask[LEFT_HAND_SLICE_39D] = True
    else:
        mask[dim // 2 :] = True
    return mask


def sustain_mask(action_dim: int) -> np.ndarray:
    dim = int(action_dim)
    mask = np.zeros((dim,), dtype=bool)
    if dim == REDUCED_ACTION_DIM:
        mask[SUSTAIN_INDEX_39D] = True
    elif dim > 0:
        mask[-1] = True
    return mask


def mask_for_goal_keys(
    goal_mask: np.ndarray,
    assignment: Mapping[int, str] | Mapping[str, Any] | None = None,
    *,
    action_dim: int | None = None,
) -> np.ndarray:
    if action_dim is None:
        action_dim = REDUCED_ACTION_DIM
    goal = np.asarray(goal_mask, dtype=np.float32).reshape(-1)
    active_keys = np.flatnonzero(goal[:88] > 0.5)
    if active_keys.size == 0:
        return full_mask(int(action_dim))
    if assignment is None:
        mask = full_mask(int(action_dim))
        mask &= ~sustain_mask(int(action_dim))
        return mask

    hands: set[str] = set()
    if "right_keys" in assignment or "left_keys" in assignment:
        right = set(int(v) for v in assignment.get("right_keys", ()))  # type: ignore[attr-defined]
        left = set(int(v) for v in assignment.get("left_keys", ()))  # type: ignore[attr-defined]
        if any(int(key) in right for key in active_keys):
            hands.add("right")
        if any(int(key) in left for key in active_keys):
            hands.add("left")
    else:
        for key in active_keys:
            hand = str(assignment.get(int(key), "")).lower()  # type: ignore[attr-defined]
            if hand in {"right", "r"}:
                hands.add("right")
            elif hand in {"left", "l"}:
                hands.add("left")
    mask = np.zeros((int(action_dim),), dtype=bool)
    if not hands or "right" in hands:
        mask |= right_hand_mask(int(action_dim))
    if not hands or "left" in hands:
        mask |= left_hand_mask(int(action_dim))
    return mask
