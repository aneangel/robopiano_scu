from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if float(denom) != 0.0 else 0.0


def _finite_rows(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    if data.ndim == 0:
        return np.asarray(np.isfinite(data), dtype=bool)
    if data.ndim == 1:
        return np.asarray(np.isfinite(data), dtype=bool)
    return np.all(np.isfinite(data), axis=tuple(range(2, data.ndim))) if data.ndim > 2 else np.all(np.isfinite(data), axis=1)


def _success_rate(values: np.ndarray, threshold: float) -> float:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite <= threshold))


def _count_valid_unassigned(unassigned_keys: np.ndarray) -> int:
    data = np.asarray(unassigned_keys)
    if data.size == 0:
        return 0
    if np.issubdtype(data.dtype, np.floating):
        valid = np.isfinite(data) & (data != -1)
    else:
        valid = data != -1
    return int(np.count_nonzero(valid))


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p95": float(np.percentile(finite, 95)),
        f"{prefix}_max": float(np.max(finite)),
    }


def _distance_summary_with_success(values: np.ndarray, prefix: str, threshold: float = 0.02) -> dict[str, float]:
    out = _summary(values, prefix)
    out[f"{prefix}_success_rate_020m"] = _success_rate(values, threshold)
    return out


def _as_scalar_float(payload: dict[str, np.ndarray], key: str, default: float) -> float:
    value = payload.get(key)
    if value is None:
        return default
    array = np.asarray(value)
    if array.size == 0:
        return default
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return default


def _anchor_row_by_frame(anchor_frames: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for row, frame in enumerate(np.asarray(anchor_frames, dtype=np.int64).reshape(-1)):
        out.setdefault(int(frame), int(row))
    return out


def _anchor_distances_for_frames(
    *,
    frames: np.ndarray,
    anchor_frames: np.ndarray,
    anchor_tips: np.ndarray,
    dense_targets: np.ndarray,
    dense_weights: np.ndarray,
    weight_min: float = 0.0,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    anchor_rows = _anchor_row_by_frame(anchor_frames)
    values: list[np.ndarray] = []
    indexed: list[tuple[int, int, float]] = []
    for sparse_row, frame in enumerate(np.asarray(frames, dtype=np.int64).reshape(-1)):
        dense_frame = int(frame)
        anchor_row = anchor_rows.get(dense_frame)
        if anchor_row is None:
            continue
        if (
            anchor_row < 0
            or anchor_row >= anchor_tips.shape[0]
            or dense_frame < 0
            or dense_frame >= dense_targets.shape[0]
            or dense_frame >= dense_weights.shape[0]
        ):
            continue
        mask = np.isfinite(dense_targets[dense_frame]).all(axis=1) & (dense_weights[dense_frame] >= float(weight_min))
        if not bool(mask.any()):
            continue
        distances = np.linalg.norm(anchor_tips[anchor_row, mask] - dense_targets[dense_frame, mask], axis=1).astype(np.float32)
        fingers = np.flatnonzero(mask).astype(np.int64)
        values.append(distances)
        indexed.extend((int(sparse_row), int(finger), float(distance)) for finger, distance in zip(fingers, distances))
    return (np.concatenate(values, axis=0) if values else np.zeros((0,), dtype=np.float32), indexed)


def _sparse_distances_by_row_and_finger(
    *,
    assignments: np.ndarray,
    fingertip_targets: np.ndarray,
    waypoint_fingertips: np.ndarray,
    sparse_rows: int,
) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    if not sparse_rows:
        return out
    sparse_target_mask = _finite_rows(fingertip_targets[:sparse_rows])
    sparse_waypoint_mask = _finite_rows(waypoint_fingertips[:sparse_rows])
    sparse_mask = (assignments[:sparse_rows] >= 0) & sparse_target_mask & sparse_waypoint_mask
    rows, fingers = np.nonzero(sparse_mask)
    for row, finger in zip(rows.astype(np.int64), fingers.astype(np.int64)):
        distance = float(
            np.linalg.norm(
                waypoint_fingertips[int(row), int(finger)] - fingertip_targets[int(row), int(finger)],
            )
        )
        out.append((int(row), int(finger), distance))
    return out


def _add_grouped_distance_summaries(
    out: dict[str, Any],
    *,
    indexed_distances: list[tuple[int, int, float]],
    active_keys_per_waypoint: np.ndarray,
    prefix: str,
) -> None:
    for finger in range(10):
        values = np.asarray([distance for _, item_finger, distance in indexed_distances if item_finger == finger], dtype=np.float32)
        out.update(_summary(values, f"{prefix}_finger_{finger}"))
    polyphonies = sorted({int(value) for value in np.asarray(active_keys_per_waypoint).reshape(-1).tolist() if int(value) > 0})
    for polyphony in polyphonies:
        values = np.asarray(
            [
                distance
                for row, _, distance in indexed_distances
                if row < active_keys_per_waypoint.shape[0] and int(active_keys_per_waypoint[row]) == polyphony
            ],
            dtype=np.float32,
        )
        out.update(_summary(values, f"{prefix}_polyphony_{polyphony}"))


def evaluate_trajectory_payload(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    target_keys = np.asarray(payload.get("target_keys", np.zeros((0, 88), dtype=np.float32)), dtype=np.float32)
    waypoint_frames = np.asarray(payload.get("waypoint_frames", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    waypoint_target_keys = np.asarray(payload.get("waypoint_target_keys", np.zeros((0, 88), dtype=np.float32)), dtype=np.float32)
    assignments = np.asarray(payload.get("assignments", np.zeros((0, 10), dtype=np.int32)), dtype=np.int32)
    unassigned_keys = np.asarray(payload.get("unassigned_keys", np.zeros((0, 10), dtype=np.int32)))
    fingertip_targets = np.asarray(payload.get("fingertip_targets", np.zeros((0, 10, 3), dtype=np.float32)), dtype=np.float32)
    waypoint_fingertips = np.asarray(payload.get("waypoint_fingertips", np.zeros((0, 10, 3), dtype=np.float32)), dtype=np.float32)

    dense_targets = np.asarray(payload.get("fingertip_trajectory_targets", np.zeros((0, 10, 3), dtype=np.float32)), dtype=np.float32)
    dense_weights = np.asarray(payload.get("fingertip_trajectory_weights", np.zeros((0, 10), dtype=np.float32)), dtype=np.float32)
    anchor_frames = np.asarray(payload.get("ik_anchor_frames_dense", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    anchor_frames_control = np.asarray(
        payload.get("ik_anchor_frames_control", np.zeros((0,), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)
    anchor_tips = np.asarray(payload.get("ik_anchor_fingertips", np.zeros((0, 10, 3), dtype=np.float32)), dtype=np.float32)
    metrics = np.asarray(payload.get("ik_anchor_metrics", np.zeros((0, 8), dtype=np.float32)), dtype=np.float32)

    qpos = np.asarray(payload.get("planned_hand_joints", np.zeros((0, 46), dtype=np.float32)), dtype=np.float32)
    velocities = np.asarray(payload.get("planned_hand_velocities", np.zeros_like(qpos)), dtype=np.float32)

    num_target_frames = int(target_keys.shape[0]) if target_keys.ndim >= 1 else 0
    num_waypoints = int(waypoint_frames.size)
    waypoint_rows = int(waypoint_target_keys.shape[0]) if waypoint_target_keys.ndim >= 2 else 0
    assignment_rows = int(assignments.shape[0]) if assignments.ndim >= 2 else 0
    sparse_rows = min(
        num_waypoints,
        waypoint_rows if waypoint_rows else num_waypoints,
        assignment_rows if assignment_rows else num_waypoints,
        int(fingertip_targets.shape[0]) if fingertip_targets.ndim >= 3 else num_waypoints,
        int(waypoint_fingertips.shape[0]) if waypoint_fingertips.ndim >= 3 else num_waypoints,
    )

    waypoint_key_mask = (waypoint_target_keys > 0) if waypoint_target_keys.ndim >= 2 else np.zeros((0, 88), dtype=bool)
    active_keys_per_waypoint = np.count_nonzero(waypoint_key_mask, axis=1) if waypoint_key_mask.size else np.zeros((0,), dtype=np.int64)
    assigned_per_waypoint = np.count_nonzero(assignments[:sparse_rows] >= 0, axis=1) if sparse_rows else np.zeros((0,), dtype=np.int64)

    num_target_key_events = int(np.count_nonzero(waypoint_key_mask))
    num_assigned_key_events = int(np.count_nonzero(assignments >= 0)) if assignments.ndim >= 2 else 0
    num_unassigned_key_events = _count_valid_unassigned(unassigned_keys)

    sparse_distance_values = np.zeros((0,), dtype=np.float32)
    sparse_indexed_distances: list[tuple[int, int, float]] = []
    if sparse_rows:
        sparse_target_mask = _finite_rows(fingertip_targets[:sparse_rows])
        sparse_waypoint_mask = _finite_rows(waypoint_fingertips[:sparse_rows])
        sparse_mask = (assignments[:sparse_rows] >= 0) & sparse_target_mask & sparse_waypoint_mask
        if bool(np.any(sparse_mask)):
            sparse_distance_values = np.linalg.norm(
                waypoint_fingertips[:sparse_rows][sparse_mask] - fingertip_targets[:sparse_rows][sparse_mask],
                axis=1,
            ).astype(np.float32)
        sparse_indexed_distances = _sparse_distances_by_row_and_finger(
            assignments=assignments,
            fingertip_targets=fingertip_targets,
            waypoint_fingertips=waypoint_fingertips,
            sparse_rows=sparse_rows,
        )

    dense_num_frames = int(dense_weights.shape[0]) if dense_weights.ndim >= 2 else int(dense_targets.shape[0]) if dense_targets.ndim >= 3 else 0
    active_dense_frames = np.any(dense_weights > 0.0, axis=1) if dense_weights.ndim >= 2 and dense_weights.size else np.zeros((dense_num_frames,), dtype=bool)
    dense_active_frame_count = int(np.count_nonzero(active_dense_frames))
    active_dense_weights = dense_weights[np.isfinite(dense_weights) & (dense_weights > 0.0)] if dense_weights.size else np.zeros((0,), dtype=np.float32)

    interpolation_substeps = int(round(_safe_div(dense_num_frames, num_target_frames))) if num_target_frames > 0 and dense_num_frames > 0 else 0
    interpolation_substeps = max(interpolation_substeps, 1) if dense_num_frames > 0 and num_target_frames > 0 else 0
    dense_waypoint_frames = (
        np.clip(waypoint_frames[:num_waypoints] * interpolation_substeps, 0, max(dense_num_frames - 1, 0)).astype(np.int64)
        if num_waypoints and dense_num_frames and interpolation_substeps
        else np.zeros((0,), dtype=np.int64)
    )
    waypoint_dense_active = (
        np.any(dense_weights[dense_waypoint_frames] > 0.0, axis=1)
        if dense_waypoint_frames.size and dense_weights.ndim >= 2 and dense_weights.shape[0] > 0
        else np.zeros((dense_waypoint_frames.size,), dtype=bool)
    )

    distances: list[np.ndarray] = []
    for row, frame in enumerate(anchor_frames):
        index = int(frame)
        if row >= anchor_tips.shape[0] or index < 0 or index >= dense_targets.shape[0] or index >= dense_weights.shape[0]:
            continue
        mask = np.isfinite(dense_targets[index]).all(axis=1) & (dense_weights[index] > 0.0)
        if bool(mask.any()):
            distances.append(np.linalg.norm(anchor_tips[row, mask] - dense_targets[index, mask], axis=1).astype(np.float32))
    distance_values = np.concatenate(distances, axis=0) if distances else np.zeros((0,), dtype=np.float32)

    control_timestep = _as_scalar_float(payload, "control_timestep", default=1.0)
    if not np.isfinite(control_timestep) or control_timestep <= 0.0:
        control_timestep = 1.0
    joint_steps = np.linalg.norm(np.diff(qpos, axis=0), axis=1) if qpos.ndim == 2 and qpos.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    joint_velocity_norm = np.linalg.norm(velocities, axis=1) if velocities.ndim == 2 else np.zeros((0,), dtype=np.float32)
    accelerations = np.diff(velocities, axis=0) / control_timestep if velocities.ndim == 2 and velocities.shape[0] > 1 else np.zeros((0, 0), dtype=np.float32)
    jerks = np.diff(accelerations, axis=0) / control_timestep if accelerations.ndim == 2 and accelerations.shape[0] > 1 else np.zeros((0, 0), dtype=np.float32)
    joint_acceleration_norm = np.linalg.norm(accelerations, axis=1) if accelerations.ndim == 2 and accelerations.size else np.zeros((0,), dtype=np.float32)
    joint_jerk_norm = np.linalg.norm(jerks, axis=1) if jerks.ndim == 2 and jerks.size else np.zeros((0,), dtype=np.float32)

    anchor_frame_set = set(int(frame) for frame in anchor_frames.tolist())
    waypoint_anchor_matches = np.asarray(
        [int(frame) in anchor_frame_set for frame in dense_waypoint_frames.tolist()],
        dtype=bool,
    )
    exact_anchor_distance_values, exact_anchor_indexed_distances = _anchor_distances_for_frames(
        frames=dense_waypoint_frames,
        anchor_frames=anchor_frames,
        anchor_tips=anchor_tips,
        dense_targets=dense_targets,
        dense_weights=dense_weights,
        weight_min=max(float(_as_scalar_float(payload, "press_weight", default=1.0)), 1.0),
    )
    weighted_anchor_distance_values, _ = _anchor_distances_for_frames(
        frames=anchor_frames,
        anchor_frames=anchor_frames,
        anchor_tips=anchor_tips,
        dense_targets=dense_targets,
        dense_weights=dense_weights,
        weight_min=1.0,
    )

    out: dict[str, Any] = {}
    out["num_target_frames"] = num_target_frames
    out["num_waypoints"] = num_waypoints
    out["num_target_key_events"] = num_target_key_events
    out["num_assigned_key_events"] = num_assigned_key_events
    out["num_unassigned_key_events"] = num_unassigned_key_events
    out["assignment_rate"] = _safe_div(num_assigned_key_events, max(num_target_key_events, 1))
    out["mean_active_keys_per_waypoint"] = float(np.mean(active_keys_per_waypoint)) if active_keys_per_waypoint.size else 0.0
    out["max_active_keys_per_waypoint"] = int(np.max(active_keys_per_waypoint)) if active_keys_per_waypoint.size else 0
    out["mean_assigned_fingers_per_waypoint"] = float(np.mean(assigned_per_waypoint)) if assigned_per_waypoint.size else 0.0
    out["max_assigned_fingers_per_waypoint"] = int(np.max(assigned_per_waypoint)) if assigned_per_waypoint.size else 0

    out.update(_summary(sparse_distance_values, "waypoint_fingertip_error"))
    out.update(_distance_summary_with_success(sparse_distance_values, "exact_waypoint_sparse_error"))
    out["exact_waypoint_sparse_success_rate_020m"] = out["exact_waypoint_sparse_error_success_rate_020m"]
    _add_grouped_distance_summaries(
        out,
        indexed_distances=sparse_indexed_distances,
        active_keys_per_waypoint=active_keys_per_waypoint,
        prefix="exact_waypoint_sparse_error",
    )
    out["waypoint_success_rate_005m"] = _success_rate(sparse_distance_values, 0.05)
    out["waypoint_success_rate_010m"] = _success_rate(sparse_distance_values, 0.10)
    out["waypoint_success_rate_020m"] = _success_rate(sparse_distance_values, 0.20)

    out["dense_num_frames"] = dense_num_frames
    out["dense_active_frame_count"] = dense_active_frame_count
    out["dense_active_frame_rate"] = _safe_div(dense_active_frame_count, dense_num_frames)
    out["dense_contact_weight_mean"] = float(np.mean(active_dense_weights)) if active_dense_weights.size else 0.0
    out["dense_contact_weight_p95"] = float(np.percentile(active_dense_weights, 95)) if active_dense_weights.size else 0.0
    out["dense_contact_weight_max"] = float(np.max(active_dense_weights)) if active_dense_weights.size else 0.0
    out["waypoint_dense_activity_rate"] = float(np.mean(waypoint_dense_active)) if waypoint_dense_active.size else 0.0
    out["waypoint_dense_inactive_count"] = int(np.count_nonzero(~waypoint_dense_active)) if waypoint_dense_active.size else num_waypoints

    out.update(_summary(distance_values, "ik_anchor_fingertip_distance"))
    out.update(_distance_summary_with_success(exact_anchor_distance_values, "exact_waypoint_anchor_error"))
    out["exact_waypoint_anchor_success_rate_020m"] = out["exact_waypoint_anchor_error_success_rate_020m"]
    out["exact_waypoint_anchor_minus_sparse_error_mean"] = (
        out["exact_waypoint_anchor_error_mean"] - out["exact_waypoint_sparse_error_mean"]
    )
    _add_grouped_distance_summaries(
        out,
        indexed_distances=exact_anchor_indexed_distances,
        active_keys_per_waypoint=active_keys_per_waypoint,
        prefix="exact_waypoint_anchor_error",
    )
    out.update(_summary(weighted_anchor_distance_values, "ik_anchor_error_weight_ge_1"))
    out["ik_anchor_success_rate_020m_weight_ge_1"] = _success_rate(weighted_anchor_distance_values, 0.02)
    out["ik_anchor_success_rate"] = float(np.mean(metrics[:, 0])) if metrics.size else 0.0
    out["ik_anchor_nfev_mean"] = float(np.mean(metrics[:, 2])) if metrics.size else 0.0
    out["ik_anchor_nfev_p95"] = float(np.percentile(metrics[:, 2], 95)) if metrics.size else 0.0
    out["num_ik_anchors"] = int(anchor_frames.size)
    out["ik_anchor_per_dense_frame_rate"] = _safe_div(anchor_frames.size, dense_num_frames)
    out["ik_anchor_per_waypoint_rate"] = _safe_div(anchor_frames.size, num_waypoints)
    out["waypoint_has_exact_anchor_rate"] = float(np.mean(waypoint_anchor_matches)) if waypoint_anchor_matches.size else 0.0
    out["waypoint_missing_exact_anchor_count"] = int(np.count_nonzero(~waypoint_anchor_matches)) if waypoint_anchor_matches.size else num_waypoints
    out["num_ik_anchor_control_frames"] = int(anchor_frames_control.size)

    out["max_joint_step"] = float(np.max(joint_steps)) if joint_steps.size else 0.0
    out["mean_joint_step"] = float(np.mean(joint_steps)) if joint_steps.size else 0.0
    out["joint_step_p95"] = float(np.percentile(joint_steps, 95)) if joint_steps.size else 0.0
    out["max_joint_velocity"] = float(np.max(joint_velocity_norm)) if joint_velocity_norm.size else 0.0
    out["mean_joint_velocity"] = float(np.mean(joint_velocity_norm)) if joint_velocity_norm.size else 0.0
    out["joint_velocity_p95"] = float(np.percentile(joint_velocity_norm, 95)) if joint_velocity_norm.size else 0.0
    out["joint_acceleration_mean"] = float(np.mean(joint_acceleration_norm)) if joint_acceleration_norm.size else 0.0
    out["joint_acceleration_p95"] = float(np.percentile(joint_acceleration_norm, 95)) if joint_acceleration_norm.size else 0.0
    out["joint_acceleration_max"] = float(np.max(joint_acceleration_norm)) if joint_acceleration_norm.size else 0.0
    out["joint_jerk_mean"] = float(np.mean(joint_jerk_norm)) if joint_jerk_norm.size else 0.0
    out["joint_jerk_p95"] = float(np.percentile(joint_jerk_norm, 95)) if joint_jerk_norm.size else 0.0
    out["joint_jerk_max"] = float(np.max(joint_jerk_norm)) if joint_jerk_norm.size else 0.0

    # Placeholders preserve schema compatibility for future rollout-integrated evaluation.
    out["online_metrics_available"] = False
    out.setdefault("missed_key_presses", None)
    out.setdefault("mispresses", None)
    out.setdefault("matched_press_events", None)
    out.setdefault("target_press_events", None)
    out.setdefault("timing_abs_error_mean_s", None)
    out.setdefault("timing_abs_error_p95_s", None)
    out.setdefault("frame_true_positives", None)
    out.setdefault("frame_false_positives", None)
    out.setdefault("frame_false_negatives", None)
    out.setdefault("frame_precision", None)
    out.setdefault("frame_recall", None)
    out.setdefault("frame_f1", None)
    return out


def evaluate_trajectory_npz(path: str | Path) -> dict[str, Any]:
    npz_path = Path(path).expanduser().resolve()
    data = np.load(npz_path, allow_pickle=False)
    return evaluate_trajectory_payload({name: data[name] for name in data.files})
