from __future__ import annotations

from typing import Any

import numpy as np


def normalize_bagatelle_metadata(
    metadata: dict[str, Any],
    *,
    q_ref_len: int | None = None,
    strict: bool = False,
    cost_weight_mode: str = "inverse",
    default_inactive_weight: float = 0.0,
) -> dict[str, Any]:
    normalized = dict(metadata)
    if not _has_bagatelle_fields(normalized):
        return normalized

    total_steps = _resolve_total_steps(normalized, q_ref_len=q_ref_len)

    desired = _normalize_desired_fingertips(normalized, total_steps=total_steps, strict=strict)
    if desired is not None:
        normalized["desired_fingertips"] = desired

    waypoint_frames = normalized.get("waypoint_frames")

    assignments = _normalize_assignment_rows(
        normalized.get("assignments"),
        total_steps=total_steps,
        waypoint_frames=waypoint_frames,
        strict=strict,
    )
    if assignments is not None:
        active_mask, inactive_mask = assignment_masks(assignments)
        normalized["active_finger_mask"] = active_mask
        normalized["inactive_finger_mask"] = inactive_mask

    weights = _normalize_weights(
        normalized.get("assignment_costs"),
        total_steps=total_steps,
        waypoint_frames=waypoint_frames,
        mode=cost_weight_mode,
        default_inactive_weight=default_inactive_weight,
        strict=strict,
    )
    if weights is not None:
        normalized["fingertip_weights"] = weights

    target_keys = _normalize_target_keys(normalized.get("target_keys"), total_steps=total_steps, strict=strict)
    if target_keys is not None:
        normalized["target_keys"] = target_keys

    if strict:
        _validate_output_shapes(normalized, total_steps=total_steps)
    return normalized


def dense_fingertip_targets_from_waypoints(
    fingertip_targets: np.ndarray,
    waypoint_frames: np.ndarray,
    total_steps: int,
    *,
    fill_mode: str = "hold",
) -> np.ndarray:
    targets = np.asarray(fingertip_targets, dtype=np.float32)
    frames = np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)
    if targets.ndim != 3 or targets.shape[1:] != (10, 3):
        raise ValueError(f"fingertip_targets must have shape [W, 10, 3], got {targets.shape}")
    if frames.shape != (targets.shape[0],):
        raise ValueError(
            f"waypoint_frames must have shape ({targets.shape[0]},), got {frames.shape}"
        )
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if fill_mode != "hold":
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")
    if targets.shape[0] == 0:
        return np.full((total_steps, 10, 3), np.nan, dtype=np.float32)

    dense = np.full((total_steps, 10, 3), np.nan, dtype=np.float32)
    bounded_frames = np.clip(frames, 0, total_steps - 1)
    last = targets[0].astype(np.float32)
    cursor = 0
    for waypoint_index, frame in enumerate(bounded_frames):
        while cursor <= int(frame) and cursor < total_steps:
            dense[cursor] = last
            cursor += 1
        last = targets[waypoint_index].astype(np.float32)
        dense[int(frame)] = last
    while cursor < total_steps:
        dense[cursor] = last
        cursor += 1
    return dense


def assignment_masks(assignments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(assignments, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 10:
        raise ValueError(f"assignments must have shape [T, 10], got {array.shape}")
    active = (array >= 0.0).astype(np.float32)
    inactive = (array < 0.0).astype(np.float32)
    return active, inactive


def _has_bagatelle_fields(metadata: dict[str, Any]) -> bool:
    return any(
        key in metadata
        for key in ("assignments", "assignment_costs", "fingertip_targets", "waypoint_fingertips")
    )


def _resolve_total_steps(metadata: dict[str, Any], *, q_ref_len: int | None) -> int | None:
    if q_ref_len is not None:
        return int(q_ref_len)
    target_keys = metadata.get("target_keys")
    if target_keys is not None:
        array = np.asarray(target_keys)
        if array.ndim >= 1:
            return int(array.shape[0])
    return None


def _normalize_desired_fingertips(
    metadata: dict[str, Any],
    *,
    total_steps: int | None,
    strict: bool,
) -> np.ndarray | None:
    if "desired_fingertips" in metadata:
        return _coerce_desired_fingertips(metadata["desired_fingertips"], strict=strict)
    if "fingertip_targets" not in metadata:
        return None
    targets = np.asarray(metadata["fingertip_targets"], dtype=np.float32)
    if targets.ndim == 2 and targets.shape == (10, 3):
        return targets.astype(np.float32)
    if targets.ndim != 3 or targets.shape[1:] != (10, 3):
        if strict:
            raise ValueError(f"fingertip_targets must have shape [T, 10, 3] or [10, 3], got {targets.shape}")
        return None

    waypoint_frames = metadata.get("waypoint_frames")
    if waypoint_frames is not None and total_steps is not None:
        frames = np.asarray(waypoint_frames)
        if frames.shape == (targets.shape[0],) and targets.shape[0] != total_steps:
            return dense_fingertip_targets_from_waypoints(targets, frames, total_steps)
    if total_steps is not None and targets.shape[0] not in (1, total_steps):
        if strict:
            raise ValueError(
                f"fingertip_targets length {targets.shape[0]} does not match total_steps {total_steps}"
            )
    return targets.astype(np.float32)


def _coerce_desired_fingertips(value: Any, *, strict: bool) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape == (10, 3):
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[1:] == (10, 3):
        return array.astype(np.float32)
    if strict:
        raise ValueError(f"desired_fingertips must have shape [T, 10, 3] or [10, 3], got {array.shape}")
    return None


def _normalize_assignment_rows(
    value: Any,
    *,
    total_steps: int | None,
    waypoint_frames: Any,
    strict: bool,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 10:
        if strict:
            raise ValueError(f"assignments must have shape [T, 10], got {array.shape}")
        return None
    if (
        total_steps is not None
        and array.shape[0] not in (1, total_steps)
        and waypoint_frames is not None
        and np.asarray(waypoint_frames).shape == (array.shape[0],)
    ):
        return _dense_rows_from_waypoints(array.astype(np.float32), np.asarray(waypoint_frames), total_steps)
    if total_steps is not None and array.shape[0] not in (1, total_steps):
        frames = np.asarray(value).shape[0]
        if strict:
            raise ValueError(f"assignments length {frames} does not match total_steps {total_steps}")
    return _broadcast_time(array.astype(np.float32), total_steps)


def _normalize_weights(
    value: Any,
    *,
    total_steps: int | None,
    waypoint_frames: Any,
    mode: str,
    default_inactive_weight: float,
    strict: bool,
) -> np.ndarray | None:
    if value is None:
        return None
    costs = np.asarray(value, dtype=np.float32)
    if costs.ndim == 1:
        costs = costs.reshape(1, -1)
    if costs.ndim != 2 or costs.shape[1] != 10:
        if strict:
            raise ValueError(f"assignment_costs must have shape [T, 10], got {costs.shape}")
        return None
    if (
        total_steps is not None
        and costs.shape[0] not in (1, total_steps)
        and waypoint_frames is not None
        and np.asarray(waypoint_frames).shape == (costs.shape[0],)
    ):
        costs = _dense_rows_from_waypoints(costs.astype(np.float32), np.asarray(waypoint_frames), total_steps)
    costs = _broadcast_time(costs.astype(np.float32), total_steps)
    if mode == "inverse":
        weights = 1.0 / np.maximum(costs, 1e-3)
        finite = np.isfinite(weights)
        if np.any(finite):
            weights = weights / max(float(np.nanmax(weights[finite])), 1e-6)
        weights[~finite] = float(default_inactive_weight)
        return weights.astype(np.float32)
    if mode == "finite_mask":
        return np.isfinite(costs).astype(np.float32)
    raise ValueError(f"Unsupported cost_weight_mode: {mode}")


def _normalize_target_keys(value: Any, *, total_steps: int | None, strict: bool) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1 and array.shape[0] == 88:
        array = array.reshape(1, 88)
    if array.ndim != 2 or array.shape[1] != 88:
        if strict:
            raise ValueError(f"target_keys must have shape [T, 88], got {array.shape}")
        return None
    return _broadcast_time(array.astype(np.float32), total_steps)


def _broadcast_time(array: np.ndarray, total_steps: int | None) -> np.ndarray:
    if total_steps is None:
        return array
    if array.shape[0] == total_steps:
        return array
    if array.shape[0] == 1:
        return np.repeat(array, total_steps, axis=0).astype(np.float32)
    return array


def _dense_rows_from_waypoints(values: np.ndarray, waypoint_frames: np.ndarray, total_steps: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    frames = np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)
    if rows.ndim != 2:
        raise ValueError(f"values must have shape [W, D], got {rows.shape}")
    if frames.shape != (rows.shape[0],):
        raise ValueError(f"waypoint_frames must have shape ({rows.shape[0]},), got {frames.shape}")
    dense = np.zeros((total_steps, rows.shape[1]), dtype=np.float32)
    if rows.shape[0] == 0:
        return dense
    bounded_frames = np.clip(frames, 0, total_steps - 1)
    last = rows[0].astype(np.float32)
    cursor = 0
    for row, frame in zip(rows, bounded_frames, strict=False):
        while cursor <= int(frame) and cursor < total_steps:
            dense[cursor] = last
            cursor += 1
        last = np.asarray(row, dtype=np.float32)
        dense[int(frame)] = last
    while cursor < total_steps:
        dense[cursor] = last
        cursor += 1
    return dense


def _validate_output_shapes(metadata: dict[str, Any], *, total_steps: int | None) -> None:
    desired = metadata.get("desired_fingertips")
    if desired is not None:
        desired_array = np.asarray(desired, dtype=np.float32)
        if desired_array.ndim == 2:
            if desired_array.shape != (10, 3):
                raise ValueError(f"desired_fingertips must have shape [10, 3], got {desired_array.shape}")
        elif desired_array.ndim == 3:
            if desired_array.shape[1:] != (10, 3):
                raise ValueError(f"desired_fingertips must have shape [T, 10, 3], got {desired_array.shape}")
            if total_steps is not None and desired_array.shape[0] != total_steps:
                raise ValueError(
                    f"desired_fingertips length {desired_array.shape[0]} does not match total_steps {total_steps}"
                )
        else:
            raise ValueError(f"desired_fingertips must have ndim 2 or 3, got {desired_array.shape}")

    for key in ("fingertip_weights", "active_finger_mask", "inactive_finger_mask"):
        value = metadata.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 10:
            raise ValueError(f"{key} must have shape [T, 10], got {array.shape}")
        if total_steps is not None and array.shape[0] != total_steps:
            raise ValueError(f"{key} length {array.shape[0]} does not match total_steps {total_steps}")

    target_keys = metadata.get("target_keys")
    if target_keys is not None:
        array = np.asarray(target_keys, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 88:
            raise ValueError(f"target_keys must have shape [T, 88], got {array.shape}")
        if total_steps is not None and array.shape[0] != total_steps:
            raise ValueError(f"target_keys length {array.shape[0]} does not match total_steps {total_steps}")
