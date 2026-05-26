from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ballade.constants import NUM_PIANO_KEYS
from ballade.interpolation import (
    build_micro_q_targets,
    finite_difference_micro_qvel,
    interpolation_phases,
)


@dataclass(frozen=True, slots=True)
class MicroTarget:
    target_q_micro: np.ndarray
    target_qvel_micro: np.ndarray
    target_q_error: np.ndarray
    target_fingertips_micro: np.ndarray | None
    goal_key_mask: np.ndarray
    press_phase: float
    time_to_press: float
    time_to_release: float
    source_frame_index: int
    microstep_index: int
    microstep_phase: float

    def as_dict(self) -> dict[str, object]:
        return {
            "target_q_micro": self.target_q_micro,
            "target_qvel_micro": self.target_qvel_micro,
            "target_q_error": self.target_q_error,
            "target_fingertips_micro": self.target_fingertips_micro,
            "goal_key_mask": self.goal_key_mask,
            "press_phase": self.press_phase,
            "time_to_press": self.time_to_press,
            "time_to_release": self.time_to_release,
            "source_frame_index": self.source_frame_index,
            "microstep_index": self.microstep_index,
            "microstep_phase": self.microstep_phase,
        }


@dataclass(frozen=True, slots=True)
class TargetSequence:
    targets: tuple[MicroTarget, ...]
    source_dt: float
    control_dt: float
    substeps: int

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self) -> Iterator[MicroTarget]:
        return iter(self.targets)

    def __getitem__(self, index: int) -> MicroTarget:
        return self.targets[index]

    @property
    def target_q_micro(self) -> np.ndarray:
        return np.stack([target.target_q_micro for target in self.targets], axis=0).astype(np.float32)

    @property
    def target_qvel_micro(self) -> np.ndarray:
        return np.stack([target.target_qvel_micro for target in self.targets], axis=0).astype(np.float32)

    @property
    def goal_key_mask(self) -> np.ndarray:
        return np.stack([target.goal_key_mask for target in self.targets], axis=0).astype(np.float32)


def _dense_goal_at(goals: np.ndarray | None, source_index: int) -> np.ndarray:
    if goals is None:
        return np.zeros((NUM_PIANO_KEYS,), dtype=np.float32)
    arr = np.asarray(goals, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"goals_20hz must be [steps, keys], got {arr.shape}")
    idx = min(max(int(source_index), 0), arr.shape[0] - 1)
    return (arr[idx, :NUM_PIANO_KEYS] > 0.5).astype(np.float32)


def _time_to_next(mask: np.ndarray, start: int, active: bool, dt: float) -> float:
    if mask.size == 0:
        return float("inf")
    desired = bool(active)
    for idx in range(start, mask.shape[0]):
        if bool(mask[idx].any()) == desired:
            return float((idx - start) * dt)
    return float("inf")


def _interpolate_optional(
    values_20hz: np.ndarray | None,
    *,
    source_dt: float,
    control_dt: float,
    method: str,
) -> np.ndarray | None:
    if values_20hz is None:
        return None
    arr = np.asarray(values_20hz, dtype=np.float32)
    if arr.ndim == 3:
        shape = arr.shape
        flat = arr.reshape(shape[0], -1)
        micro = build_micro_q_targets(flat, source_dt=source_dt, control_dt=control_dt, method=method)
        return micro.reshape(micro.shape[0], micro.shape[1], *shape[1:]).astype(np.float32)
    return build_micro_q_targets(arr, source_dt=source_dt, control_dt=control_dt, method=method)


def build_micro_targets(
    q_20hz: np.ndarray,
    goals_20hz: np.ndarray | None = None,
    qvel_20hz: np.ndarray | None = None,
    fingertips_20hz: np.ndarray | None = None,
    source_dt: float = 0.05,
    control_dt: float = 0.005,
    method: str = "linear",
) -> TargetSequence:
    q = np.asarray(q_20hz, dtype=np.float32)
    q_micro = build_micro_q_targets(q, qvel_20hz=qvel_20hz, source_dt=source_dt, control_dt=control_dt, method=method)
    if qvel_20hz is not None:
        qvel_micro = build_micro_q_targets(
            qvel_20hz,
            source_dt=source_dt,
            control_dt=control_dt,
            method="linear",
        )
    else:
        qvel_micro = finite_difference_micro_qvel(q_micro, q_initial=q[0], control_dt=control_dt)
    fingertip_micro = _interpolate_optional(
        fingertips_20hz,
        source_dt=source_dt,
        control_dt=control_dt,
        method="linear",
    )
    phases = interpolation_phases(source_dt, control_dt)
    substeps = int(phases.size)
    dense_goal_masks = []
    for source_index in range(q_micro.shape[0]):
        for _micro in range(substeps):
            dense_goal_masks.append(_dense_goal_at(goals_20hz, source_index + 1))
    dense_goal_mask_arr = np.stack(dense_goal_masks, axis=0).astype(np.float32)

    targets = []
    flat_index = 0
    for source_index in range(q_micro.shape[0]):
        for micro_index, phase in enumerate(phases):
            goal_mask = dense_goal_mask_arr[flat_index]
            active = bool(goal_mask.any())
            time_to_press = 0.0 if active else _time_to_next(dense_goal_mask_arr, flat_index, True, control_dt)
            time_to_release = 0.0 if not active else _time_to_next(dense_goal_mask_arr, flat_index, False, control_dt)
            targets.append(
                MicroTarget(
                    target_q_micro=q_micro[source_index, micro_index].copy(),
                    target_qvel_micro=qvel_micro[source_index, micro_index].copy(),
                    target_q_error=np.zeros_like(q_micro[source_index, micro_index], dtype=np.float32),
                    target_fingertips_micro=(
                        None
                        if fingertip_micro is None
                        else np.asarray(fingertip_micro[source_index, micro_index], dtype=np.float32).copy()
                    ),
                    goal_key_mask=goal_mask.copy(),
                    press_phase=float(phase) if active else 0.0,
                    time_to_press=float(time_to_press),
                    time_to_release=float(time_to_release),
                    source_frame_index=int(source_index),
                    microstep_index=int(micro_index),
                    microstep_phase=float(phase),
                )
            )
            flat_index += 1
    return TargetSequence(
        targets=tuple(targets),
        source_dt=float(source_dt),
        control_dt=float(control_dt),
        substeps=int(substeps),
    )
