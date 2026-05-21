from __future__ import annotations

import numpy as np


def compute_fingertip_assignment_metrics(
    current_fingertips: np.ndarray,
    desired_fingertips: np.ndarray,
    *,
    active_finger_mask: np.ndarray | None = None,
    contact_mask: np.ndarray | None = None,
    clearance_axis: int = 2,
    clearance_margin: float = 0.01,
) -> dict[str, float]:
    current = _coerce_fingertips(current_fingertips, "current_fingertips")
    desired = _coerce_fingertips(desired_fingertips, "desired_fingertips")
    if current is None or desired is None:
        return {}
    steps = min(current.shape[0], desired.shape[0])
    if steps == 0:
        return {}
    current = current[:steps]
    desired = desired[:steps]
    finite_target = np.isfinite(desired).all(axis=-1)
    active = _coerce_mask(active_finger_mask, steps=steps, default=finite_target)
    active = np.logical_and(active > 0.5, finite_target)

    errors = np.linalg.norm(np.where(finite_target[..., None], current - desired, 0.0), axis=-1)
    active_errors = errors[active]
    if active_errors.size == 0:
        return {}

    metrics = {
        "fingertip/active_l2_mean": float(np.mean(active_errors)),
        "fingertip/active_l2_p95": float(np.percentile(active_errors, 95)),
        "fingertip/active_l2_max": float(np.max(active_errors)),
        "fingertip/assigned_success_rate_at_1cm": float(np.mean(active_errors <= 0.01)),
        "fingertip/assigned_success_rate_at_2cm": float(np.mean(active_errors <= 0.02)),
        "fingertip/assigned_success_rate_at_5cm": float(np.mean(active_errors <= 0.05)),
    }

    contact = _coerce_mask(contact_mask, steps=steps, default=active)
    contact_active = np.logical_and(contact > 0.5, active)
    contact_errors = errors[contact_active]
    metrics["fingertip/contact_window_l2_mean"] = (
        float(np.mean(contact_errors)) if contact_errors.size else metrics["fingertip/active_l2_mean"]
    )

    inactive = np.logical_and(np.logical_not(active), np.isfinite(current).all(axis=-1))
    if np.any(inactive):
        clearance = current[..., clearance_axis]
        violations = np.logical_and(inactive, clearance < float(clearance_margin))
        metrics["fingertip/inactive_clearance_violation_rate"] = float(np.mean(violations[inactive]))
    else:
        metrics["fingertip/inactive_clearance_violation_rate"] = 0.0
    return metrics


def _coerce_fingertips(value: np.ndarray, name: str) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return None
    if array.ndim == 2 and array.shape == (10, 3):
        return array[None, ...].astype(np.float32)
    if array.ndim == 3 and array.shape[1:] == (10, 3):
        return array.astype(np.float32)
    raise ValueError(f"{name} must have shape [T, 10, 3] or [10, 3], got {array.shape}")


def _coerce_mask(value: np.ndarray | None, *, steps: int, default: np.ndarray) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1 and array.shape[0] == 10:
        array = np.repeat(array[None, :], steps, axis=0)
    if array.ndim != 2 or array.shape[1] != 10:
        raise ValueError(f"mask must have shape [T, 10] or [10], got {array.shape}")
    return array[:steps].astype(np.float32)
