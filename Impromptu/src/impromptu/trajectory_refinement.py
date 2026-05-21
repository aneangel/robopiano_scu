from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from impromptu.config import ImpromptuConfig


def _clip_qpos(kin: Any, qpos: np.ndarray) -> np.ndarray:
    if hasattr(kin, "clip_qpos"):
        return np.asarray(kin.clip_qpos(qpos), dtype=np.float32)
    lower = np.asarray(getattr(kin, "joint_lower", np.full(qpos.shape[-1], -np.inf)), dtype=np.float32)
    upper = np.asarray(getattr(kin, "joint_upper", np.full(qpos.shape[-1], np.inf)), dtype=np.float32)
    return np.clip(np.asarray(qpos, dtype=np.float32), lower, upper).astype(np.float32)


def refine_dense_qpos_trajectory(
    *,
    kin: Any,
    dense_qpos: np.ndarray,
    fingertip_targets: np.ndarray,
    fingertip_weights: np.ndarray,
    neutral_qpos: np.ndarray,
    config: ImpromptuConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Optionally refine dense qpos chunks against fingertip targets.

    This is deliberately small and CPU-bound: it uses SciPy least-squares on
    short overlapping-free chunks and is disabled by default. Anchor IK remains
    the primary dense planner path.
    """
    qpos = np.asarray(dense_qpos, dtype=np.float32)
    targets = np.asarray(fingertip_targets, dtype=np.float32)
    weights = np.asarray(fingertip_weights, dtype=np.float32)
    if not bool(config.enable_trajectory_refinement):
        return qpos.copy(), {"enabled": False, "chunks": 0, "optimizer_success_count": 0, "nfev_sum": 0}
    if qpos.ndim != 2 or qpos.shape[0] < 3 or targets.shape[:2] != weights.shape or targets.shape[0] != qpos.shape[0]:
        return qpos.copy(), {"enabled": True, "skipped": "invalid_shapes", "chunks": 0, "optimizer_success_count": 0, "nfev_sum": 0}

    total, dim = qpos.shape
    window = max(int(config.trajectory_refinement_window_frames), 3)
    lower = np.tile(np.asarray(kin.joint_lower, dtype=np.float64), window)
    upper = np.tile(np.asarray(kin.joint_upper, dtype=np.float64), window)
    neutral = _clip_qpos(kin, neutral_qpos).astype(np.float64)
    refined = qpos.copy()
    success_count = 0
    nfev_sum = 0
    chunks = 0

    for start in range(0, total, window):
        end = min(start + window, total)
        if end - start < 3:
            continue
        chunks += 1
        original = refined[start:end].astype(np.float64, copy=True)
        chunk_targets = targets[start:end]
        chunk_weights = weights[start:end]
        active = np.isfinite(chunk_targets).all(axis=2) & (chunk_weights > 0.0)
        local_lower = lower[: (end - start) * dim]
        local_upper = upper[: (end - start) * dim]

        def residual(values: np.ndarray) -> np.ndarray:
            rows = values.reshape(end - start, dim)
            parts: list[np.ndarray] = []
            for row in range(rows.shape[0]):
                q = _clip_qpos(kin, rows[row])
                fingertips = np.asarray(kin.fingertip_positions_for_qpos(q), dtype=np.float32)
                mask = active[row]
                if bool(mask.any()):
                    err = (fingertips[mask] - chunk_targets[row, mask]) * chunk_weights[row, mask, None]
                    parts.append(err.reshape(-1).astype(np.float64) * float(config.trajectory_refinement_fingertip_weight))
            clipped_rows = np.stack([_clip_qpos(kin, row) for row in rows], axis=0).astype(np.float64)
            parts.append(np.diff(clipped_rows, axis=0).reshape(-1) * float(config.trajectory_refinement_velocity_weight))
            if clipped_rows.shape[0] > 2:
                parts.append(np.diff(clipped_rows, n=2, axis=0).reshape(-1) * float(config.trajectory_refinement_acceleration_weight))
            if clipped_rows.shape[0] > 3:
                parts.append(np.diff(clipped_rows, n=3, axis=0).reshape(-1) * float(config.trajectory_refinement_jerk_weight))
            parts.append((clipped_rows - neutral[None, :]).reshape(-1) * float(config.trajectory_refinement_neutral_weight))
            endpoint_delta = np.concatenate([clipped_rows[0] - original[0], clipped_rows[-1] - original[-1]], axis=0)
            parts.append(endpoint_delta * float(config.trajectory_refinement_endpoint_weight))
            return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float64)

        opt = least_squares(
            residual,
            original.reshape(-1),
            bounds=(local_lower, local_upper),
            max_nfev=max(int(config.trajectory_refinement_max_nfev), 1),
            ftol=float(config.ik_ftol),
            xtol=float(config.ik_xtol),
            gtol=float(config.ik_gtol),
        )
        refined[start:end] = np.stack(
            [_clip_qpos(kin, row) for row in opt.x.reshape(end - start, dim)],
            axis=0,
        ).astype(np.float32)
        success_count += int(bool(opt.success))
        nfev_sum += int(opt.nfev)

    return refined.astype(np.float32), {
        "enabled": True,
        "chunks": int(chunks),
        "optimizer_success_count": int(success_count),
        "nfev_sum": int(nfev_sum),
    }
