from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np

from ballade.features import as_float_array, get_field, target_fingertip_error, target_q_error


def _weight(weights: Mapping[str, float] | Any | None, name: str, default: float) -> float:
    if weights is None:
        return float(default)
    if isinstance(weights, Mapping):
        return float(weights.get(name, default))
    if is_dataclass(weights):
        return float(asdict(weights).get(name, default))
    return float(getattr(weights, name, default))


def _mean_square(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.square(arr)))


def hand_q_tracking_cost(obs: Any, target: Any) -> float:
    return _mean_square(target_q_error(obs, target))


def hand_qvel_tracking_cost(obs: Any, target: Any) -> float:
    qvel = as_float_array(get_field(obs, "qvel", "hand_qvel"))
    target_qvel = as_float_array(get_field(target, "target_qvel_micro", "qvel"))
    width = min(qvel.size, target_qvel.size)
    if width <= 0:
        return 0.0
    return _mean_square(target_qvel[:width] - qvel[:width])


def fingertip_tracking_cost(obs: Any, target: Any) -> float:
    return _mean_square(target_fingertip_error(obs, target))


def keypress_cost(obs: Any, target: Any, *, target_weight: float = 1.0, non_target_weight: float = 1.0) -> float:
    piano = as_float_array(get_field(obs, "piano_activation", "piano_state"))
    goal = as_float_array(get_field(target, "goal_key_mask", "goals"))
    width = min(piano.size, goal.size, 88)
    if width <= 0:
        return 0.0
    active = np.clip(piano[:width], 0.0, 1.0)
    desired = (goal[:width] > 0.5).astype(np.float32)
    miss = desired * np.square(1.0 - active)
    false = (1.0 - desired) * np.square(active)
    return float(target_weight * miss.mean() + non_target_weight * false.mean())


def action_smoothness_cost(action: np.ndarray, previous_action: np.ndarray | None) -> float:
    if previous_action is None:
        return 0.0
    action_arr = as_float_array(action)
    prev_arr = as_float_array(previous_action)
    width = min(action_arr.size, prev_arr.size)
    if width <= 0:
        return 0.0
    return _mean_square(action_arr[:width] - prev_arr[:width])


def action_saturation_cost(action: np.ndarray) -> float:
    action_arr = np.abs(as_float_array(action))
    if action_arr.size == 0:
        return 0.0
    excess = np.maximum(action_arr - 0.95, 0.0) / 0.05
    return float(np.mean(np.square(excess)))


def total_tracking_cost(
    obs: Any,
    target: Any,
    action: np.ndarray,
    previous_action: np.ndarray | None,
    weights: Mapping[str, float] | Any | None,
) -> float:
    phase = float(get_field(target, "microstep_phase", default=0.0) or 0.0)
    endpoint_boost = 1.0 if phase < 0.999 else _weight(weights, "hand_endpoint_q", 4.0)
    key_weight = _weight(weights, "keypress", 8.0)
    non_target_key_weight = _weight(weights, "non_target_keypress", 5.0)
    total = 0.0
    total += endpoint_boost * _weight(weights, "hand_q", 1.0) * hand_q_tracking_cost(obs, target)
    total += _weight(weights, "hand_qvel", 0.25) * hand_qvel_tracking_cost(obs, target)
    total += _weight(weights, "fingertip", 3.0) * fingertip_tracking_cost(obs, target)
    total += keypress_cost(obs, target, target_weight=key_weight, non_target_weight=non_target_key_weight)
    total += _weight(weights, "action_smoothness", 0.05) * action_smoothness_cost(action, previous_action)
    total += _weight(weights, "action_saturation", 0.02) * action_saturation_cost(action)
    return float(total)
