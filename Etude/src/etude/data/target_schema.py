from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PHASE_LABELS: tuple[str, ...] = (
    "approach",
    "pre_contact",
    "contact",
    "hold",
    "release",
    "recovery",
)

_PHASE_ALIASES: dict[str, str] = {
    "approach": "approach",
    "attack": "approach",
    "precontact": "pre_contact",
    "pre_contact": "pre_contact",
    "contact": "contact",
    "press": "contact",
    "hold": "hold",
    "sustain": "hold",
    "release": "release",
    "recovery": "recovery",
    "recover": "recovery",
    "idle": "recovery",
    "rest": "recovery",
    "unknown": "recovery",
}


def standardize_controller_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    q_ref: np.ndarray | None = None,
    qdot_ref: np.ndarray | None = None,
    horizon: int | None = None,
    dt: float | None = None,
    key_dim: int = 88,
    lookahead_steps: Sequence[int] = (1, 5, 10),
    active_threshold: float = 0.5,
) -> dict[str, Any]:
    """Return canonical controller metadata for goal-oriented rollouts.

    The schema keeps the dense trajectory references separate from piano goals:
    ``q_ref``/``qdot_ref`` are tracking priors, while ``target_keys`` plus the
    derived timing/phase fields describe the press objective. Existing aliases
    are accepted so older datasets and configs keep loading.
    """

    source = dict(metadata or {})
    result = dict(source)
    if q_ref is None:
        q_ref = _first_present(source, ("q_ref", "q_ref_dense"))
    if qdot_ref is None:
        qdot_ref = _first_present(source, ("qdot_ref", "qvel_ref"))
    if q_ref is not None:
        result["q_ref"] = np.asarray(q_ref, dtype=np.float32)
    if qdot_ref is not None:
        result["qdot_ref"] = np.asarray(qdot_ref, dtype=np.float32)
    resolved_horizon = _resolve_horizon(horizon, q_ref, source)
    resolved_dt = float(dt if dt is not None else source.get("dt", 0.005))
    result["dt"] = resolved_dt

    target_keys = _first_present(
        source,
        (
            "target_keys",
            "target_keys_now",
            "target_key_targets",
            "key_targets",
            "desired_keys",
        ),
    )
    if target_keys is not None:
        target_matrix = _coerce_time_matrix(
            target_keys,
            feature_dim=key_dim,
            horizon=resolved_horizon,
            name="target_keys",
        )
        result["target_keys"] = target_matrix
        result["target_keys_now"] = target_matrix
        result["target_key_lookahead"] = build_target_key_lookahead(
            target_matrix,
            lookahead_steps=lookahead_steps,
        )
        if "time_to_next_press" not in result and "time_to_next_active_key" not in result:
            time_to_next = compute_time_to_next_press(
                target_matrix,
                dt=resolved_dt,
                active_threshold=active_threshold,
            )
            result["time_to_next_press"] = time_to_next
            result["time_to_next_active_key"] = time_to_next

    desired_fingertips = _first_present(
        source,
        ("desired_fingertips", "fingertip_ref", "target_fingertip_positions"),
    )
    if desired_fingertips is not None:
        result["desired_fingertips"] = np.asarray(desired_fingertips, dtype=np.float32)
        result.setdefault("fingertip_ref", result["desired_fingertips"])

    active_mask = _first_present(source, ("active_finger_mask", "active_fingers"))
    if active_mask is not None:
        result["active_finger_mask"] = np.asarray(active_mask, dtype=np.float32)

    phases = _first_present(source, ("phase_schedule", "phases", "phase_labels"))
    if phases is None and target_keys is not None:
        phases = infer_phase_schedule(result["target_keys"], active_threshold=active_threshold)
    if phases is not None:
        phase_schedule = _coerce_phase_schedule(phases, horizon=resolved_horizon)
        result["phase_schedule"] = phase_schedule
        result["phases"] = phase_schedule
    return result


def metadata_at_timestep(metadata: Mapping[str, Any], t: int) -> dict[str, Any]:
    """Extract per-step goal fields while retaining sequence fields."""

    index = max(int(t), 0)
    output: dict[str, Any] = {}
    if "target_keys" in metadata:
        target = np.asarray(metadata["target_keys"], dtype=np.float32)
        output["target_keys"] = target
        output["target_keys_now"] = timestep_value(target, index)
    if "target_key_lookahead" in metadata:
        output["target_key_lookahead"] = timestep_value(metadata["target_key_lookahead"], index)
    for key in ("time_to_next_press", "time_to_next_active_key"):
        if key in metadata:
            output[key] = scalar_timestep_value(metadata[key], index)
    for key in (
        "desired_fingertips",
        "fingertip_ref",
        "active_finger_mask",
        "inactive_finger_mask",
        "contact_mask",
    ):
        if key in metadata:
            output[key] = timestep_value(metadata[key], index)
    phase_schedule = metadata.get("phase_schedule", metadata.get("phases"))
    if phase_schedule is not None:
        output["phase"] = canonicalize_phase(timestep_value(phase_schedule, index))
    return output


def build_target_key_lookahead(
    target_keys: np.ndarray,
    *,
    lookahead_steps: Sequence[int],
) -> np.ndarray:
    target = _coerce_time_matrix(target_keys, feature_dim=target_keys.shape[-1], name="target_keys")
    steps = tuple(int(step) for step in lookahead_steps)
    if not steps:
        return np.zeros((target.shape[0], 0, target.shape[1]), dtype=np.float32)
    indices = np.arange(target.shape[0], dtype=np.int64)[:, None] + np.asarray(steps, dtype=np.int64)[None, :]
    indices = np.clip(indices, 0, target.shape[0] - 1)
    return target[indices].astype(np.float32)


def compute_time_to_next_press(
    target_keys: np.ndarray,
    *,
    dt: float,
    active_threshold: float = 0.5,
) -> np.ndarray:
    target = _coerce_time_matrix(target_keys, feature_dim=target_keys.shape[-1], name="target_keys")
    active = np.any(target >= float(active_threshold), axis=1)
    output = np.zeros(target.shape[0], dtype=np.float32)
    next_active = -1
    for index in range(target.shape[0] - 1, -1, -1):
        if active[index]:
            next_active = index
        output[index] = 0.0 if next_active < 0 else float(next_active - index) * float(dt)
    return output


def infer_phase_schedule(
    target_keys: np.ndarray,
    *,
    active_threshold: float = 0.5,
) -> np.ndarray:
    target = _coerce_time_matrix(target_keys, feature_dim=target_keys.shape[-1], name="target_keys")
    active = np.any(target >= float(active_threshold), axis=1)
    phases: list[str] = []
    for index, is_active in enumerate(active.tolist()):
        prev_active = bool(active[index - 1]) if index > 0 else False
        next_active = bool(active[index + 1]) if index + 1 < active.size else False
        if is_active and prev_active:
            phases.append("hold")
        elif is_active:
            phases.append("contact")
        elif prev_active:
            phases.append("release")
        elif next_active:
            phases.append("approach")
        else:
            phases.append("recovery")
    return np.asarray(phases, dtype=object)


def phase_to_one_hot(phase: Any, *, phase_labels: Sequence[str] = PHASE_LABELS) -> np.ndarray:
    labels = tuple(phase_labels)
    one_hot = np.zeros(len(labels), dtype=np.float32)
    key = canonicalize_phase(phase)
    try:
        one_hot[labels.index(key)] = 1.0
    except ValueError:
        if "recovery" in labels:
            one_hot[labels.index("recovery")] = 1.0
    return one_hot


def canonicalize_phase(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return canonicalize_phase(value.item())
        if value.size == 1:
            return canonicalize_phase(value.reshape(-1)[0])
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _PHASE_ALIASES.get(key, key if key in PHASE_LABELS else "recovery")


def timestep_value(value: Any, t: int) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.ndim == 1:
        if array.dtype.kind in {"U", "S", "O"}:
            return array[_clip_index(t, array.shape[0])]
        return array.astype(np.float32)
    if array.ndim == 2 and array.shape == (10, 3):
        return array.astype(np.float32)
    return array[_clip_index(t, array.shape[0])].astype(np.float32)


def scalar_timestep_value(value: Any, t: int) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(array[_clip_index(t, array.size)])


def _first_present(source: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return None


def _resolve_horizon(horizon: int | None, q_ref: Any, source: Mapping[str, Any]) -> int | None:
    if horizon is not None:
        return int(horizon)
    if q_ref is not None:
        array = np.asarray(q_ref)
        if array.ndim >= 2:
            return int(array.shape[0])
    for key in ("target_keys", "desired_fingertips", "active_finger_mask", "phase_schedule", "phases"):
        value = source.get(key)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim >= 1:
            return int(array.shape[0])
    return None


def _coerce_time_matrix(
    value: Any,
    *,
    feature_dim: int,
    horizon: int | None = None,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        if array.shape[0] != feature_dim:
            raise ValueError(f"{name} must have feature dimension {feature_dim}, got {array.shape}")
        repeats = 1 if horizon is None else max(int(horizon), 1)
        array = np.repeat(array[None, :], repeats, axis=0)
    if array.ndim != 2 or array.shape[1] != feature_dim:
        raise ValueError(f"{name} must have shape [T, {feature_dim}], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one timestep")
    return array.astype(np.float32)


def _coerce_phase_schedule(value: Any, *, horizon: int | None) -> np.ndarray:
    array = np.asarray(value, dtype=object).reshape(-1)
    if array.size == 0:
        raise ValueError("phase schedule must contain at least one entry")
    if horizon is not None and array.size == 1 and int(horizon) > 1:
        array = np.repeat(array, int(horizon))
    if horizon is not None and array.size != int(horizon):
        raise ValueError(f"phase schedule must have length {horizon}, got {array.size}")
    return np.asarray([canonicalize_phase(item) for item in array.tolist()], dtype=object)


def _clip_index(t: int, horizon: int) -> int:
    return int(np.clip(int(t), 0, max(int(horizon) - 1, 0)))
