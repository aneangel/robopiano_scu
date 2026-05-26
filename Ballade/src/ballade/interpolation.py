from __future__ import annotations

import math

import numpy as np


def _as_2d(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [steps, dim], got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} needs at least two source steps")
    return arr


def _substeps(source_dt: float, control_dt: float) -> int:
    if float(source_dt) <= 0.0 or float(control_dt) <= 0.0:
        raise ValueError("source_dt and control_dt must be positive")
    ratio = float(source_dt) / float(control_dt)
    rounded = int(round(ratio))
    if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("source_dt must be an integer multiple of control_dt")
    return rounded


def interpolation_phases(source_dt: float = 0.05, control_dt: float = 0.005) -> np.ndarray:
    """Return endpoint-inclusive phases [1/substeps, ..., 1.0]."""
    steps = _substeps(source_dt, control_dt)
    return (np.arange(1, steps + 1, dtype=np.float32) / float(steps)).astype(np.float32)


def linear_interpolate(q0: np.ndarray, q1: np.ndarray, phase: float | np.ndarray) -> np.ndarray:
    q0_arr = np.asarray(q0, dtype=np.float32)
    q1_arr = np.asarray(q1, dtype=np.float32)
    phase_arr = np.asarray(phase, dtype=np.float32)
    return ((1.0 - phase_arr) * q0_arr + phase_arr * q1_arr).astype(np.float32)


def hermite_interpolate(
    q0: np.ndarray,
    q1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    phase: float | np.ndarray,
    dt: float,
) -> np.ndarray:
    q0_arr = np.asarray(q0, dtype=np.float32)
    q1_arr = np.asarray(q1, dtype=np.float32)
    v0_arr = np.asarray(v0, dtype=np.float32)
    v1_arr = np.asarray(v1, dtype=np.float32)
    t = np.asarray(phase, dtype=np.float32)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return (h00 * q0_arr + h10 * float(dt) * v0_arr + h01 * q1_arr + h11 * float(dt) * v1_arr).astype(
        np.float32
    )


def build_micro_q_targets(
    q_20hz: np.ndarray,
    qvel_20hz: np.ndarray | None = None,
    source_dt: float = 0.05,
    control_dt: float = 0.005,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate source hand states into 200 Hz micro targets.

    The returned shape is [source_steps - 1, substeps, q_dim]. Phases are
    endpoint-inclusive: 1/substeps, 2/substeps, ..., 1.0. With 10 substeps, the
    final micro target of interval k equals source waypoint k + 1.
    """
    q = _as_2d("q_20hz", q_20hz)
    phases = interpolation_phases(source_dt, control_dt)
    method_value = str(method).lower()
    if method_value not in {"linear", "hermite"}:
        raise ValueError(f"Unknown interpolation method: {method}")
    qvel = None
    if method_value == "hermite":
        if qvel_20hz is None:
            raise ValueError("qvel_20hz is required for Hermite interpolation")
        qvel = _as_2d("qvel_20hz", qvel_20hz)
        if qvel.shape != q.shape:
            raise ValueError(f"qvel_20hz shape {qvel.shape} must match q_20hz shape {q.shape}")

    intervals = []
    for idx in range(q.shape[0] - 1):
        if method_value == "linear":
            micro = np.stack([linear_interpolate(q[idx], q[idx + 1], phase) for phase in phases], axis=0)
        else:
            assert qvel is not None
            micro = np.stack(
                [
                    hermite_interpolate(q[idx], q[idx + 1], qvel[idx], qvel[idx + 1], phase, source_dt)
                    for phase in phases
                ],
                axis=0,
            )
        intervals.append(micro.astype(np.float32))
    return np.stack(intervals, axis=0).astype(np.float32)


def finite_difference_micro_qvel(
    target_q_micro: np.ndarray,
    q_initial: np.ndarray,
    control_dt: float = 0.005,
) -> np.ndarray:
    micro = np.asarray(target_q_micro, dtype=np.float32)
    if micro.ndim != 3:
        raise ValueError(f"target_q_micro must be [intervals, substeps, dim], got {micro.shape}")
    initial = np.asarray(q_initial, dtype=np.float32).reshape(-1)
    flat = micro.reshape(-1, micro.shape[-1])
    previous = np.vstack([initial[None, :], flat[:-1]])
    return ((flat - previous) / float(control_dt)).reshape(micro.shape).astype(np.float32)
