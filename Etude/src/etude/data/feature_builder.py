from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from etude.data.target_schema import (
    PHASE_LABELS,
    phase_to_one_hot,
    scalar_timestep_value,
    standardize_controller_metadata,
    timestep_value,
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    lookahead_steps: tuple[int, ...] = (1, 5, 10)
    include_target_keys: bool = True
    include_fingertips: bool = True
    target_key_lookahead_steps: tuple[int, ...] = ()
    include_time_to_next_press: bool = False
    include_phase: bool = False
    phase_labels: tuple[str, ...] = PHASE_LABELS

    def __post_init__(self) -> None:
        if any(int(step) < 0 for step in self.lookahead_steps):
            raise ValueError("lookahead_steps must be non-negative")
        if any(int(step) < 0 for step in self.target_key_lookahead_steps):
            raise ValueError("target_key_lookahead_steps must be non-negative")


def build_tracking_features(
    *,
    q: np.ndarray,
    qdot: np.ndarray,
    q_ref: np.ndarray,
    qdot_ref: np.ndarray,
    t: int,
    previous_action: np.ndarray,
    target_keys: np.ndarray | None = None,
    fingertips: np.ndarray | None = None,
    metadata: dict | None = None,
    spec: FeatureSpec | None = None,
) -> np.ndarray:
    spec = spec or FeatureSpec()
    q = _vector(q, name="q")
    qdot = _vector(qdot, name="qdot")
    previous_action = _vector(previous_action, name="previous_action")
    q_ref = _matrix(q_ref, name="q_ref")
    qdot_ref = _matrix(qdot_ref, name="qdot_ref")
    if q.shape[0] != q_ref.shape[1]:
        raise ValueError("q dimension must match q_ref feature dimension")
    if qdot.shape != q.shape:
        raise ValueError("qdot must match q")
    if qdot_ref.shape != q_ref.shape:
        raise ValueError("qdot_ref must match q_ref")
    index = _bounded_index(t, q_ref.shape[0])
    pieces: list[np.ndarray] = [q, qdot, q_ref[index], qdot_ref[index]]
    pieces.extend(q_ref[_bounded_index(index + step, q_ref.shape[0])] for step in spec.lookahead_steps)
    pieces.append(previous_action)
    schema = standardize_controller_metadata(
        metadata,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        horizon=q_ref.shape[0],
        lookahead_steps=spec.target_key_lookahead_steps or spec.lookahead_steps,
    )
    if target_keys is None:
        target_keys = schema.get("target_keys")
    if spec.include_target_keys and target_keys is not None:
        target_matrix = _matrix(target_keys, name="target_keys")
        pieces.append(target_matrix[_bounded_index(index, target_matrix.shape[0])])
        pieces.extend(
            target_matrix[_bounded_index(index + step, target_matrix.shape[0])]
            for step in spec.target_key_lookahead_steps
        )
    if spec.include_time_to_next_press:
        time_to_next = schema.get("time_to_next_press", schema.get("time_to_next_active_key"))
        value = 0.0 if time_to_next is None else scalar_timestep_value(time_to_next, index)
        pieces.append(np.asarray([value], dtype=np.float32))
    if spec.include_phase:
        phases = schema.get("phase_schedule", schema.get("phases"))
        phase = None if phases is None else timestep_value(phases, index)
        pieces.append(phase_to_one_hot(phase, phase_labels=spec.phase_labels))
    if spec.include_fingertips and fingertips is not None:
        fingertip_array = np.asarray(fingertips, dtype=np.float32)
        if fingertip_array.ndim == 1:
            pieces.append(fingertip_array)
        else:
            fingertip_matrix = _matrix(fingertips, name="fingertips")
            pieces.append(fingertip_matrix[_bounded_index(index, fingertip_matrix.shape[0])])
    return _concat_float32(pieces)


def _vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [T, D], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one row")
    return array


def _bounded_index(index: int, length: int) -> int:
    return int(np.clip(index, 0, length - 1))


def _concat_float32(parts: Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]).astype(np.float32)
