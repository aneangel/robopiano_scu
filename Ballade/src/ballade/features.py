from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any

import numpy as np


def get_field(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if is_dataclass(obj):
        available = {field.name for field in fields(obj)}
        for name in names:
            if name in available:
                return getattr(obj, name)
    return default


def as_float_array(value: Any, *, default_width: int = 0) -> np.ndarray:
    if value is None:
        return np.zeros((int(default_width),), dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def pad_or_trim(values: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if int(width) <= 0:
        return np.zeros((0,), dtype=np.float32)
    out = np.zeros((int(width),), dtype=np.float32)
    copy_width = min(out.size, arr.size)
    if copy_width > 0:
        out[:copy_width] = arr[:copy_width]
    return out


def target_q_error(obs: Any, target: Any) -> np.ndarray:
    q = as_float_array(get_field(obs, "q", "hand_qpos"))
    target_q = as_float_array(get_field(target, "target_q_micro", "q"))
    width = min(q.size, target_q.size)
    if width <= 0:
        return np.zeros((0,), dtype=np.float32)
    return (target_q[:width] - q[:width]).astype(np.float32)


def target_fingertip_error(obs: Any, target: Any) -> np.ndarray:
    fingertips = as_float_array(get_field(obs, "fingertips", "hand_fingertips"))
    target_tips = as_float_array(get_field(target, "target_fingertips_micro", "fingertips"))
    width = min(fingertips.size, target_tips.size)
    if width <= 0:
        return np.zeros((0,), dtype=np.float32)
    return (target_tips[:width] - fingertips[:width]).astype(np.float32)


def build_feature_vector(
    obs: Any,
    target: Any,
    previous_action: np.ndarray | None = None,
    future_targets: list[Any] | tuple[Any, ...] | None = None,
    *,
    future_window: int = 0,
) -> np.ndarray:
    q = as_float_array(get_field(obs, "q", "hand_qpos"))
    qvel = as_float_array(get_field(obs, "qvel", "hand_qvel"), default_width=q.size)
    fingertips = as_float_array(get_field(obs, "fingertips", "hand_fingertips"))
    piano = as_float_array(get_field(obs, "piano_activation", "piano_state"))
    prev = as_float_array(previous_action)
    target_q = as_float_array(get_field(target, "target_q_micro", "q"))
    target_qvel = as_float_array(get_field(target, "target_qvel_micro", "qvel"), default_width=target_q.size)
    q_error = target_q_error(obs, target)
    tip_error = target_fingertip_error(obs, target)
    goal_mask = as_float_array(get_field(target, "goal_key_mask", "goals"))
    scalars = np.asarray(
        [
            float(get_field(target, "press_phase", default=0.0) or 0.0),
            float(get_field(target, "time_to_press", default=0.0) or 0.0),
            float(get_field(target, "time_to_release", default=0.0) or 0.0),
            float(get_field(target, "microstep_phase", default=0.0) or 0.0),
        ],
        dtype=np.float32,
    )
    future_parts: list[np.ndarray] = []
    if future_targets and int(future_window) > 0:
        for future in list(future_targets)[: int(future_window)]:
            future_parts.append(as_float_array(get_field(future, "target_q_micro", "q")))
            future_parts.append(as_float_array(get_field(future, "goal_key_mask", "goals")))
    parts = [
        q,
        qvel,
        fingertips,
        piano,
        prev,
        target_q,
        target_qvel,
        q_error,
        tip_error,
        goal_mask,
        scalars,
        *future_parts,
    ]
    nonempty = [part.reshape(-1).astype(np.float32) for part in parts if part.size]
    if not nonempty:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(nonempty, axis=0).astype(np.float32)
