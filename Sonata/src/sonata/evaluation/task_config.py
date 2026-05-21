from __future__ import annotations

from typing import Any

import numpy as np

NORMALIZED_MINUS_ONE_TO_ONE = "normalized_minus_one_to_one"
ACTUATOR_UNITS = "actuator_units"
REDUCED_ACTION_DIM = 39


def build_rollout_task_kwargs(
    *,
    control_timestep: float,
    expected_action_dim: int,
    reduced_action_space: bool | None = None,
) -> dict[str, Any]:
    if reduced_action_space is None:
        reduced_action_space = int(expected_action_dim) == REDUCED_ACTION_DIM
    return {
        "control_timestep": float(control_timestep),
        "n_steps_lookahead": 1,
        "reduced_action_space": bool(reduced_action_space),
    }


def rollout_action_source_scale(rollout_config: dict[str, Any] | None = None) -> str:
    return str((rollout_config or {}).get("action_source_scale", NORMALIZED_MINUS_ONE_TO_ONE))


def adapt_action_to_spec(
    action: np.ndarray,
    action_spec: Any,
    *,
    source_scale: str = NORMALIZED_MINUS_ONE_TO_ONE,
) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
    control = np.zeros_like(minimum, dtype=np.float32)
    width = min(int(control.size), int(values.size))
    if width <= 0:
        return control
    if source_scale == NORMALIZED_MINUS_ONE_TO_ONE:
        clipped = np.clip(values[:width], -1.0, 1.0)
        control[:width] = minimum[:width] + 0.5 * (clipped + 1.0) * (maximum[:width] - minimum[:width])
    elif source_scale == ACTUATOR_UNITS:
        control[:width] = values[:width]
    else:
        raise ValueError(f"Unsupported rollout action source scale: {source_scale!r}.")
    return np.clip(control, minimum, maximum)


def validate_rollout_action_dim(
    *,
    actual_action_dim: int,
    expected_action_dim: int,
    environment_name: str,
    require_exact: bool = True,
) -> None:
    actual = int(actual_action_dim)
    expected = int(expected_action_dim)
    if require_exact and actual != expected:
        raise ValueError(
            "RoboPianist environment "
            f"`{environment_name}` exposes action_dim={actual}, but Sonata expected "
            f"action_dim={expected}. Use reduced_action_space=True for 39-D RP1M/Sonata actions."
        )
    if not require_exact and actual < expected:
        raise ValueError(
            "RoboPianist environment "
            f"`{environment_name}` exposes action_dim={actual}, which is smaller than the "
            f"primitive prior action_dim={expected}."
        )
