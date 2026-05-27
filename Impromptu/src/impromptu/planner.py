from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from impromptu.config import ImpromptuConfig
from impromptu.anchor_selection import select_ik_anchor_frames
from impromptu.fingertip_trajectory import NUM_FINGERS, FingertipTrajectory, build_fingertip_trajectory
from impromptu.ik_solver import IK_ANCHOR_METRIC_COLUMNS, interpolate_anchor_qpos, solve_fingertip_trajectory_anchors
from impromptu.joint_space_trajectory import (
    TRAJECTORY_MODE_DENSE_FINGERTIP_IK,
    TRAJECTORY_MODE_JOINT_SPACE_STRAIGHTEN,
    TRAJECTORY_MODES,
    build_joint_space_straightened_trajectory,
)
from impromptu.paths import ensure_repo_paths
from impromptu.trajectory_refinement import refine_dense_qpos_trajectory
from impromptu.trajectory import ImpromptuTrajectory

ensure_repo_paths()
from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import HAND_STATE_DIM, BagatelleKinematics, IKResult  # noqa: E402
from bagatelle.planner import IK_METRIC_COLUMNS as BAGATELLE_IK_METRIC_COLUMNS  # noqa: E402
from bagatelle.planner import plan_target_keys as plan_bagatelle_target_keys  # noqa: E402
from intermezzo.keys import validate_target_keys  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402


def _bagatelle_config(config: ImpromptuConfig) -> BagatelleConfig:
    """Return the Bagatelle configuration used for all assignment and IK.

    Impromptu owns dense anchor selection and trajectory IK, while Bagatelle
    still owns sparse key-to-finger assignment and press-pose seeds. Forward
    the Impromptu CLI knobs that affect those Bagatelle responsibilities so a
    single planning command has coherent assignment and seed-IK behavior.
    """
    return BagatelleConfig(
        control_timestep=float(config.control_timestep),
        threshold=float(config.threshold),
        environment_name=str(config.environment_name),
        seed=int(config.seed),
        reduced_action_space=bool(config.reduced_action_space),
        ik_fingertip_weight=float(config.ik_fingertip_weight),
        ik_key_front_weight=float(config.ik_key_front_weight),
        ik_key_width_weight=float(config.ik_key_width_weight),
        ik_key_height_weight=float(config.ik_key_height_weight),
        ik_smoothness_weight=float(config.ik_smoothness_weight),
        ik_neutral_weight=float(config.ik_neutral_weight),
        ik_inactive_fingertip_clearance_weight=float(config.ik_inactive_fingertip_clearance_weight),
        ik_inactive_fingertip_clearance=float(config.ik_inactive_fingertip_clearance),
        ik_unassigned_fingertip_strategy=str(config.ik_unassigned_fingertip_strategy),
        ik_unassigned_fingertip_avoidance_weight=float(config.ik_unassigned_fingertip_avoidance_weight),
        ik_unassigned_fingertip_avoidance_radius=float(config.ik_unassigned_fingertip_avoidance_radius),
        ik_wrong_key_xy_avoidance_weight=float(config.ik_wrong_key_xy_avoidance_weight),
        ik_wrong_key_xy_avoidance_radius=float(config.ik_wrong_key_xy_avoidance_radius),
        ik_max_nfev=int(config.ik_max_nfev),
        ik_ftol=float(config.ik_ftol),
        ik_xtol=float(config.ik_xtol),
        ik_gtol=float(config.ik_gtol),
        residual_success_threshold=float(config.residual_success_threshold),
        ik_multistart_on_failure=bool(config.ik_multistart_on_failure),
        ik_multistart_seed_count=int(config.ik_multistart_seed_count),
        ik_multistart_forearm_tx_grid=int(config.ik_multistart_forearm_tx_grid),
        ik_static_contact_validation=bool(config.ik_static_contact_validation),
        ik_static_contact_settle_steps=int(config.ik_static_contact_settle_steps),
        ik_static_contact_wrong_key_weight=float(config.ik_static_contact_wrong_key_weight),
        ik_static_contact_missed_key_weight=float(config.ik_static_contact_missed_key_weight),
        ik_static_contact_residual_weight=float(config.ik_static_contact_residual_weight),
        ik_static_contact_failure_weight=float(config.ik_static_contact_failure_weight),
        ik_cache_mode=str(getattr(config, "ik_cache_mode", "off")),
        ik_cache_jaccard_threshold=float(getattr(config, "ik_cache_jaccard_threshold", 0.8)),
        rhapsody_ik_enabled=bool(config.rhapsody_ik_enabled),
        rhapsody_ik_checkpoint=str(config.rhapsody_ik_checkpoint),
        rhapsody_ik_refinement_steps=int(config.rhapsody_ik_refinement_steps),
        rhapsody_ik_refinement_lr=float(config.rhapsody_ik_refinement_lr),
        rhapsody_ik_device=str(config.rhapsody_ik_device),
        rhapsody_ik_candidate_scoring=bool(config.rhapsody_ik_candidate_scoring),
        rhapsody_ik_coordinate_transform=str(config.rhapsody_ik_coordinate_transform),
        rhapsody_ik_y_offset=float(config.rhapsody_ik_y_offset),
        rhapsody_ik_fill_inactive_from_previous=bool(config.rhapsody_ik_fill_inactive_from_previous),
        rhapsody_ik_seed_max_active_error=float(config.rhapsody_ik_seed_max_active_error),
        rhapsody_ik_seed_require_previous_improvement=bool(config.rhapsody_ik_seed_require_previous_improvement),
        key_target_front_offset=float(config.key_target_front_offset),
        key_target_top_offset=float(config.key_target_top_offset),
        key_press_depth=float(config.key_press_depth),
        distance_weight=float(config.assignment_distance_weight),
        same_finger_bonus=float(config.same_finger_bonus),
        reassignment_penalty=float(config.reassignment_penalty),
        finger_crossing_penalty=float(config.finger_crossing_penalty),
        wrong_hand_penalty=float(config.wrong_hand_penalty),
        wrong_hand_split_key=int(config.wrong_hand_split_key),
        assignment_dynamic_hand_split=bool(config.assignment_dynamic_hand_split),
        assignment_dynamic_hand_split_min_span=int(config.assignment_dynamic_hand_split_min_span),
        assignment_dynamic_hand_split_min_keys=int(config.assignment_dynamic_hand_split_min_keys),
        large_jump_penalty=float(config.large_jump_penalty),
        same_key_same_finger_bonus=float(config.same_key_same_finger_bonus),
        assignment_strategy=str(config.assignment_strategy),
        assignment_distance_weight=float(config.assignment_distance_weight),
        assignment_hand_zone_weight=float(config.assignment_hand_zone_weight),
        assignment_finger_zone_weight=float(config.assignment_finger_zone_weight),
        assignment_crossing_weight=float(config.finger_crossing_penalty),
        assignment_hold_weight=float(config.assignment_hold_weight),
        assignment_reach_weight=float(config.assignment_reach_weight),
        assignment_black_key_weight=float(config.assignment_black_key_weight),
        assignment_hard_hand_split=bool(config.assignment_hard_hand_split),
        assignment_middle_key=int(config.assignment_middle_key),
        assignment_wrong_hand_penalty=float(config.wrong_hand_penalty),
        assignment_reach_soft_limit=float(config.assignment_reach_soft_limit),
        assignment_top_k=int(config.assignment_top_k),
        assignment_top_k_extra_penalty=float(config.assignment_top_k_extra_penalty),
        assignment_beam_width=int(config.assignment_beam_width),
        assignment_candidates_per_step=int(config.assignment_candidates_per_step),
        assignment_fail_if_unassigned=bool(config.assignment_fail_if_unassigned),
        assignment_unassigned_penalty=float(config.assignment_unassigned_penalty),
        assignment_ik_residual_weight=float(config.assignment_ik_residual_weight),
        assignment_ik_max_residual_weight=float(config.assignment_ik_max_residual_weight),
        assignment_ik_failure_penalty=float(config.assignment_ik_failure_penalty),
        assignment_motion_weight=float(config.assignment_motion_weight),
    )


def _polyphony_stats(target_keys: np.ndarray, *, threshold: float) -> dict[str, object]:
    keys = validate_target_keys(target_keys)
    active_counts = np.count_nonzero(keys[:, :88] > float(threshold), axis=1).astype(np.int32)
    active_frames = active_counts > 0
    return {
        "active_frame_rate": float(np.mean(active_frames)) if active_counts.size else 0.0,
        "mean_polyphony": float(np.mean(active_counts)) if active_counts.size else 0.0,
        "active_mean_polyphony": float(np.mean(active_counts[active_frames])) if bool(np.any(active_frames)) else 0.0,
        "max_polyphony": int(np.max(active_counts)) if active_counts.size else 0,
    }


def _with_adaptive_complex_song_defaults(
    config: ImpromptuConfig,
    target_keys: np.ndarray,
) -> tuple[ImpromptuConfig, dict[str, object]]:
    stats = _polyphony_stats(target_keys, threshold=float(config.threshold))
    active_mean = float(stats["active_mean_polyphony"])
    max_polyphony = int(stats["max_polyphony"])
    is_complex = bool(
        active_mean >= float(config.adaptive_active_mean_polyphony_threshold)
        or max_polyphony >= int(config.adaptive_max_polyphony_threshold)
    )
    metadata: dict[str, object] = {
        **stats,
        "enabled": bool(config.adaptive_complex_song_defaults),
        "activated": False,
        "active_mean_polyphony_threshold": float(config.adaptive_active_mean_polyphony_threshold),
        "max_polyphony_threshold": int(config.adaptive_max_polyphony_threshold),
        "overrides": {},
    }
    if not bool(config.adaptive_complex_song_defaults) or not is_complex:
        return config, metadata

    defaults = ImpromptuConfig(
        adaptive_complex_song_defaults=bool(config.adaptive_complex_song_defaults),
        adaptive_active_mean_polyphony_threshold=float(config.adaptive_active_mean_polyphony_threshold),
        adaptive_max_polyphony_threshold=int(config.adaptive_max_polyphony_threshold),
    )
    desired = {
        "key_press_depth": 0.0040,
        "wrong_hand_penalty": 4.0,
        "wrong_hand_split_key": 48,
        "assignment_dynamic_hand_split": True,
        "assignment_dynamic_hand_split_min_span": 12,
        "assignment_dynamic_hand_split_min_keys": 3,
        "finger_crossing_penalty": 1.0,
        "same_key_same_finger_bonus": 0.25,
        "ik_unassigned_fingertip_strategy": "avoid_mispresses",
        "ik_unassigned_fingertip_avoidance_weight": 0.5,
        "ik_unassigned_fingertip_avoidance_radius": 0.03,
    }
    overrides: dict[str, object] = {}
    for name, value in desired.items():
        if getattr(config, name) == getattr(defaults, name):
            overrides[name] = value
    if not overrides:
        metadata["activated"] = True
        return config, metadata
    metadata["activated"] = True
    metadata["overrides"] = dict(overrides)
    return replace(config, **overrides), metadata


def _metadata(
    *,
    config: ImpromptuConfig,
    target_keys: np.ndarray,
    waypoint_frames: np.ndarray,
    dense_total: int,
    anchor_frames: np.ndarray,
    anchor_results: tuple[IKResult, ...],
    kin: Any,
) -> dict[str, Any]:
    nfev = np.asarray([result.nfev for result in anchor_results], dtype=np.float32)
    max_residuals = np.asarray([result.max_residual for result in anchor_results], dtype=np.float32)
    dense_dt = float(config.control_timestep) / float(max(int(config.interpolation_substeps), 1))
    return {
        "planner": "impromptu_dense_anchor_ik_assignment",
        "target_keys_shape": list(target_keys.shape),
        "num_waypoints": int(waypoint_frames.size),
        "waypoint_frames": waypoint_frames.astype(int).tolist(),
        "control_timestep": float(config.control_timestep),
        "dense_control_timestep": dense_dt,
        "interpolation_substeps": int(config.interpolation_substeps),
        "num_dense_frames": int(dense_total),
        "num_ik_anchor_frames": int(anchor_frames.size),
        "ik_anchor_fraction": float(anchor_frames.size / dense_total) if dense_total else 0.0,
        "ik_success_count": int(sum(bool(result.success) for result in anchor_results)),
        "ik_optimizer_success_count": int(sum(bool(result.optimizer_success) for result in anchor_results)),
        "ik_nfev_mean": float(nfev.mean()) if nfev.size else 0.0,
        "ik_nfev_p95": float(np.percentile(nfev, 95)) if nfev.size else 0.0,
        "ik_max_residual_mean": float(max_residuals.mean()) if max_residuals.size else 0.0,
        "ik_max_residual_p95": float(np.percentile(max_residuals, 95)) if max_residuals.size else 0.0,
        "environment_name": str(getattr(kin, "environment_name", config.environment_name)),
        "midi_proto_path": str(getattr(kin, "midi_proto_path", "")),
        "output_root": str(config.output_root),
        "config": config.to_dict(),
        "ik_anchor_metric_columns": list(IK_ANCHOR_METRIC_COLUMNS),
    }


def _metric_row_for_anchor_pose(
    *,
    fingertips: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> np.ndarray:
    mask = np.isfinite(targets).all(axis=1) & (weights > 0.0)
    active_count = int(np.count_nonzero(mask))
    if active_count:
        distances = np.linalg.norm(fingertips[mask] - targets[mask], axis=1).astype(np.float32)
    else:
        distances = np.zeros((0,), dtype=np.float32)
    mean_distance = float(np.mean(distances)) if distances.size else 0.0
    max_distance = float(np.max(distances)) if distances.size else 0.0
    residual_norm = float(np.linalg.norm(distances)) if distances.size else 0.0
    return np.asarray(
        [
            float(max_distance <= float(threshold)),
            1.0,
            0.0,
            0.5 * residual_norm * residual_norm,
            mean_distance,
            max_distance,
            residual_norm,
            float(active_count),
        ],
        dtype=np.float32,
    )


def _control_frame_anchor_metrics(
    *,
    anchor_frames: np.ndarray,
    anchor_fingertips: np.ndarray,
    fingertip_targets: np.ndarray,
    fingertip_weights: np.ndarray,
    config: ImpromptuConfig,
) -> np.ndarray:
    rows = []
    for row, dense_frame in enumerate(anchor_frames.astype(np.int64).reshape(-1)):
        if 0 <= int(dense_frame) < fingertip_targets.shape[0]:
            rows.append(
                _metric_row_for_anchor_pose(
                    fingertips=anchor_fingertips[row],
                    targets=fingertip_targets[int(dense_frame)],
                    weights=fingertip_weights[int(dense_frame)],
                    threshold=float(config.residual_success_threshold),
                )
            )
        else:
            rows.append(np.zeros((len(IK_ANCHOR_METRIC_COLUMNS),), dtype=np.float32))
    if rows:
        return np.stack(rows, axis=0).astype(np.float32)
    return np.zeros((0, len(IK_ANCHOR_METRIC_COLUMNS)), dtype=np.float32)


def _waypoint_release_frames(target_keys: np.ndarray, waypoint_frames: np.ndarray, *, threshold: float) -> np.ndarray:
    keys = np.asarray(target_keys, dtype=np.float32)
    frames = np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)
    if frames.size == 0 or keys.size == 0:
        return np.zeros((0,), dtype=np.int64)
    active = keys[:, :88] > float(threshold)
    releases: list[int] = []
    total = int(active.shape[0])
    for frame in frames:
        start = int(np.clip(int(frame), 0, max(total - 1, 0)))
        row = active[start]
        end = start
        while end + 1 < total and np.array_equal(active[end + 1], row):
            end += 1
        releases.append(int(end))
    return np.asarray(releases, dtype=np.int64)


def _anchor_results_for_qpos_rows(
    *,
    anchor_frames: np.ndarray,
    anchor_qpos: np.ndarray,
    anchor_fingertips: np.ndarray,
    fingertip_targets: np.ndarray,
    fingertip_weights: np.ndarray,
    config: ImpromptuConfig,
    message: str,
) -> tuple[IKResult, ...]:
    results: list[IKResult] = []
    for row, dense_frame in enumerate(np.asarray(anchor_frames, dtype=np.int64).reshape(-1)):
        if 0 <= int(dense_frame) < fingertip_targets.shape[0]:
            targets = np.asarray(fingertip_targets[int(dense_frame)], dtype=np.float32)
            weights = np.asarray(fingertip_weights[int(dense_frame)], dtype=np.float32)
        else:
            targets = np.full((NUM_FINGERS, 3), np.nan, dtype=np.float32)
            weights = np.zeros((NUM_FINGERS,), dtype=np.float32)
        mask = np.isfinite(targets).all(axis=1) & (weights > 0.0)
        active_fingers = np.flatnonzero(mask).astype(np.int32)
        if active_fingers.size:
            distances = np.linalg.norm(
                anchor_fingertips[row, active_fingers.astype(np.int64)] - targets[active_fingers.astype(np.int64)],
                axis=1,
            ).astype(np.float32)
        else:
            distances = np.zeros((0,), dtype=np.float32)
        max_residual = float(np.max(distances)) if distances.size else 0.0
        residual_norm = float(np.linalg.norm(distances)) if distances.size else 0.0
        results.append(
            IKResult(
                pose=np.asarray(anchor_qpos[row], dtype=np.float32).copy(),
                fingertip_positions=np.asarray(anchor_fingertips[row], dtype=np.float32).copy(),
                assigned_distances=distances,
                residual_norm=residual_norm,
                max_residual=max_residual,
                success=bool(max_residual <= float(config.residual_success_threshold)),
                optimizer_success=True,
                optimizer_status=0,
                optimizer_message=message,
                optimizer_cost=0.0,
                nfev=0,
                active_keys=np.zeros((0,), dtype=np.int32),
                assigned_keys=np.full((active_fingers.size,), -1, dtype=np.int32),
                assigned_finger_indices=active_fingers,
                unassigned_keys=np.zeros((0,), dtype=np.int32),
            )
        )
    return tuple(results)


def _fingertips_for_qpos_rows(kin: Any, qpos_rows: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos_rows, dtype=np.float32)
    if qpos.size == 0:
        return np.zeros((0, NUM_FINGERS, 3), dtype=np.float32)
    return np.stack(
        [np.asarray(kin.fingertip_positions_for_qpos(row), dtype=np.float32) for row in qpos],
        axis=0,
    ).astype(np.float32)


def plan_target_keys(
    target_keys: np.ndarray,
    config: ImpromptuConfig | None = None,
    *,
    kinematics: BagatelleKinematics | None = None,
) -> ImpromptuTrajectory:
    base_cfg = config or ImpromptuConfig()
    keys = validate_target_keys(target_keys)
    cfg, adaptive_metadata = _with_adaptive_complex_song_defaults(base_cfg, keys)
    bag_cfg = _bagatelle_config(cfg)

    owns_kinematics = kinematics is None
    kin = kinematics or BagatelleKinematics(bag_cfg, target_keys=keys)
    try:
        bagatelle_plan = plan_bagatelle_target_keys(keys, config=bag_cfg, kinematics=kin)

        substeps = max(int(cfg.interpolation_substeps), 1)
        if bagatelle_plan.waypoint_frames.size:
            fingertip_traj = build_fingertip_trajectory(
                total_steps=int(keys.shape[0]),
                waypoint_frames=bagatelle_plan.waypoint_frames,
                assignments=bagatelle_plan.assignments,
                fingertip_targets=bagatelle_plan.fingertip_targets,
                waypoint_fingertips=bagatelle_plan.waypoint_fingertips,
                config=cfg,
            )
        else:
            dense_total_empty = int(keys.shape[0]) * substeps
            fingertip_traj = FingertipTrajectory(
                targets=np.full((dense_total_empty, NUM_FINGERS, 3), np.nan, dtype=np.float32),
                weights=np.zeros((dense_total_empty, NUM_FINGERS), dtype=np.float32),
                dense_frames=np.arange(dense_total_empty, dtype=np.int64),
            )

        dense_total = int(fingertip_traj.targets.shape[0])
        bagatelle_control_qpos = np.asarray(bagatelle_plan.planned_hand_joints, dtype=np.float32).copy()
        if bagatelle_control_qpos.shape[0] != keys.shape[0]:
            raise RuntimeError(
                f"Bagatelle planned_hand_joints has {bagatelle_control_qpos.shape[0]} frames, expected {keys.shape[0]}"
            )
        control_anchor_frames = (np.arange(keys.shape[0], dtype=np.int64) * substeps).astype(np.int64)
        bagatelle_dense_seed, _ = interpolate_anchor_qpos(
            anchor_frames=control_anchor_frames,
            anchor_qpos=bagatelle_control_qpos,
            dense_total=dense_total,
        )
        waypoint_frames_dense = np.clip(
            np.asarray(bagatelle_plan.waypoint_frames, dtype=np.int64) * substeps,
            0,
            max(dense_total - 1, 0),
        ).astype(np.int64)
        neutral_qpos = np.asarray(getattr(kin, "neutral_qpos", np.zeros((HAND_STATE_DIM,), dtype=np.float32)), dtype=np.float32)
        trajectory_mode = str(cfg.trajectory_mode)
        if trajectory_mode not in TRAJECTORY_MODES:
            raise ValueError(f"trajectory_mode must be one of {TRAJECTORY_MODES}, got {trajectory_mode!r}")

        joint_space_metadata: dict[str, object] = {}
        if trajectory_mode == TRAJECTORY_MODE_JOINT_SPACE_STRAIGHTEN:
            joint_space_plan = build_joint_space_straightened_trajectory(
                total_steps=int(keys.shape[0]),
                waypoint_frames=bagatelle_plan.waypoint_frames,
                waypoint_release_frames=_waypoint_release_frames(
                    keys,
                    bagatelle_plan.waypoint_frames,
                    threshold=float(cfg.threshold),
                ),
                waypoint_qpos=bagatelle_plan.waypoint_hand_joints,
                assignments=bagatelle_plan.assignments,
                neutral_qpos=neutral_qpos,
                config=cfg,
                kinematics=kin,
            )
            anchor_frames = joint_space_plan.anchor_frames
            anchor_solution_qpos = joint_space_plan.anchor_qpos
            waypoint_hand_joints_out = joint_space_plan.waypoint_qpos
            planned_dense = joint_space_plan.dense_qpos
            segment_ids_dense = joint_space_plan.segment_ids
            joint_space_metadata = dict(joint_space_plan.metadata)
            anchor_solution_tips = _fingertips_for_qpos_rows(kin, anchor_solution_qpos)
            anchor_metrics = _control_frame_anchor_metrics(
                anchor_frames=anchor_frames,
                anchor_fingertips=anchor_solution_tips,
                fingertip_targets=fingertip_traj.targets,
                fingertip_weights=fingertip_traj.weights,
                config=cfg,
            )
            anchor_results = _anchor_results_for_qpos_rows(
                anchor_frames=anchor_frames,
                anchor_qpos=anchor_solution_qpos,
                anchor_fingertips=anchor_solution_tips,
                fingertip_targets=fingertip_traj.targets,
                fingertip_weights=fingertip_traj.weights,
                config=cfg,
                message="joint-space straight-finger anchor; no dense fingertip IK solve",
            )
            refinement_metadata = {
                "enabled": False,
                "skipped": "joint_space_straighten_mode",
                "chunks": 0,
                "optimizer_success_count": 0,
                "nfev_sum": 0,
            }
        elif trajectory_mode == TRAJECTORY_MODE_DENSE_FINGERTIP_IK:
            waypoint_hand_joints_out = bagatelle_plan.waypoint_hand_joints
            anchor_frames = select_ik_anchor_frames(
                fingertip_targets=fingertip_traj.targets,
                fingertip_weights=fingertip_traj.weights,
                waypoint_frames_dense=waypoint_frames_dense,
                config=cfg,
            )
            anchor_seed_qpos = (
                bagatelle_dense_seed[anchor_frames]
                if anchor_frames.size
                else np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
            )
            anchor_solution = solve_fingertip_trajectory_anchors(
                kin=kin,
                fingertip_targets=fingertip_traj.targets,
                fingertip_weights=fingertip_traj.weights,
                anchor_frames=anchor_frames,
                initial_qpos=anchor_seed_qpos,
                neutral_qpos=neutral_qpos,
                config=cfg,
            )
            anchor_solution_qpos = anchor_solution.qpos
            anchor_solution_tips = anchor_solution.fingertips
            anchor_metrics = anchor_solution.metrics
            anchor_results = anchor_solution.results
            if bool(cfg.preserve_waypoint_press_qpos) and anchor_solution_qpos.size and waypoint_frames_dense.size:
                blend = float(np.clip(float(cfg.waypoint_qpos_blend), 0.0, 1.0))
                if blend > 0.0:
                    anchor_solution_qpos = anchor_solution_qpos.astype(np.float32, copy=True)
                    for waypoint_row, dense_frame in enumerate(waypoint_frames_dense.astype(np.int64)):
                        rows = np.flatnonzero(anchor_frames == int(dense_frame))
                        if rows.size and waypoint_row < bagatelle_plan.waypoint_hand_joints.shape[0]:
                            row = int(rows[0])
                            seed = np.asarray(bagatelle_plan.waypoint_hand_joints[waypoint_row], dtype=np.float32)
                            anchor_solution_qpos[row] = (
                                (1.0 - blend) * anchor_solution_qpos[row] + blend * seed
                            ).astype(np.float32)
                    anchor_solution_tips = _fingertips_for_qpos_rows(kin, anchor_solution_qpos)
                    anchor_metrics = _control_frame_anchor_metrics(
                        anchor_frames=anchor_frames,
                        anchor_fingertips=anchor_solution_tips,
                        fingertip_targets=fingertip_traj.targets,
                        fingertip_weights=fingertip_traj.weights,
                        config=cfg,
                    )
            planned_dense, segment_ids_dense = interpolate_anchor_qpos(
                anchor_frames=anchor_frames,
                anchor_qpos=anchor_solution_qpos,
                dense_total=dense_total,
            )
            refinement_metadata: dict[str, object] = {
                "enabled": False,
                "chunks": 0,
                "optimizer_success_count": 0,
                "nfev_sum": 0,
            }
            planned_dense, refinement_metadata = refine_dense_qpos_trajectory(
                kin=kin,
                dense_qpos=planned_dense,
                fingertip_targets=fingertip_traj.targets,
                fingertip_weights=fingertip_traj.weights,
                neutral_qpos=neutral_qpos,
                config=cfg,
            )
        else:
            raise AssertionError(f"unhandled trajectory mode: {trajectory_mode}")
        segment_ids = segment_ids_dense[::substeps][: keys.shape[0]].astype(np.int32, copy=True)
        if segment_ids.shape[0] < keys.shape[0]:
            pad_value = int(segment_ids[-1]) if segment_ids.size else -1
            segment_ids = np.concatenate(
                [segment_ids, np.full((keys.shape[0] - segment_ids.shape[0],), pad_value, dtype=np.int32)],
                axis=0,
            )
        dense_dt = float(cfg.control_timestep) / float(substeps)
        dense_velocities = compute_hand_velocities(planned_dense, control_timestep=dense_dt)
        planned = planned_dense[::substeps][: keys.shape[0]].astype(np.float32, copy=True)
        if planned.shape[0] < keys.shape[0]:
            pad = planned[-1:] if planned.size else np.zeros((1, HAND_STATE_DIM), dtype=np.float32)
            planned = np.concatenate([planned, np.repeat(pad, keys.shape[0] - planned.shape[0], axis=0)], axis=0)
        planned_velocities = compute_hand_velocities(planned, control_timestep=float(cfg.control_timestep))

        metadata = _metadata(
            config=cfg,
            target_keys=keys,
            waypoint_frames=bagatelle_plan.waypoint_frames,
            dense_total=dense_total,
            anchor_frames=anchor_frames,
            anchor_results=anchor_results,
            kin=kin,
        )
        metadata["bagatelle_planner"] = bagatelle_plan.metadata.get("planner", "bagatelle")
        metadata["bagatelle_assignment_mode"] = bagatelle_plan.metadata.get("assignment_mode")
        metadata["bagatelle_config"] = bagatelle_plan.metadata.get("config", bag_cfg.to_dict())
        if "keyset_cache_report" in bagatelle_plan.metadata:
            metadata["keyset_cache_report"] = bagatelle_plan.metadata["keyset_cache_report"]
        for _static_key in (
            "static_contact_hits_total",
            "static_contact_wrongs_total",
            "static_contact_misses_total",
            "static_contact_played_total",
        ):
            if _static_key in bagatelle_plan.metadata:
                metadata[_static_key] = bagatelle_plan.metadata[_static_key]
        metadata["trajectory_mode"] = trajectory_mode
        metadata["assignment_source"] = "Bagatelle.plan_target_keys"
        if trajectory_mode == TRAJECTORY_MODE_JOINT_SPACE_STRAIGHTEN:
            metadata["planner"] = "impromptu_joint_space_straighten_assignment"
            metadata["ik_source"] = "Bagatelle.solve_press_pose_sparse_only"
            metadata["planned_hand_joints_source"] = "downsampled_from_joint_space_straight_finger_trajectory"
            metadata["dense_hand_joints_source"] = "joint_space_straight_finger_interpolation"
            metadata["dense_ik_seed_source"] = "not_used_joint_space_straighten_mode"
            metadata["waypoint_hand_joints_source"] = "bagatelle_press_qpos_with_idle_fingers_straightened"
            metadata["fingertip_attraction_forces"] = "disabled_by_joint_space_mode"
            metadata["joint_space_trajectory"] = joint_space_metadata
        else:
            metadata["ik_source"] = "Impromptu.solve_fingertip_trajectory_anchors"
            metadata["planned_hand_joints_source"] = "downsampled_from_dense_anchor_ik"
            metadata["dense_hand_joints_source"] = "interpolated_from_impromptu_dense_ik_anchors"
            metadata["dense_ik_seed_source"] = "interpolated_bagatelle_control_frame_qpos"
            metadata["waypoint_hand_joints_source"] = "bagatelle_press_qpos"
        metadata["selected_anchor_config"] = {
            "anchor_stride": int(cfg.anchor_stride),
            "solve_contact_window_only": bool(cfg.solve_contact_window_only),
            "include_midpoint_anchors": bool(cfg.include_midpoint_anchors),
            "anchor_change_threshold": float(cfg.anchor_change_threshold),
        }
        metadata["adaptive_complex_song_defaults"] = adaptive_metadata
        metadata["trajectory_refinement"] = refinement_metadata
        metadata["sparse_press_ik_metric_columns"] = list(BAGATELLE_IK_METRIC_COLUMNS)
        metadata["sparse_press_ik_metrics_shape"] = list(bagatelle_plan.ik_metrics.shape)
        metadata["sparse_press_ik_success_count"] = int(np.count_nonzero(bagatelle_plan.ik_metrics[:, 0])) if bagatelle_plan.ik_metrics.size else 0

        return ImpromptuTrajectory(
            target_keys=keys,
            waypoint_frames=bagatelle_plan.waypoint_frames,
            waypoint_target_keys=bagatelle_plan.waypoint_target_keys,
            assignments=bagatelle_plan.assignments,
            assignment_costs=bagatelle_plan.assignment_costs,
            fingertip_targets=bagatelle_plan.fingertip_targets,
            waypoint_fingertips=bagatelle_plan.waypoint_fingertips,
            unassigned_keys=bagatelle_plan.unassigned_keys,
            fingertip_trajectory_targets=fingertip_traj.targets,
            fingertip_trajectory_weights=fingertip_traj.weights,
            fingertip_trajectory_dense_frames=fingertip_traj.dense_frames,
            ik_anchor_frames_dense=anchor_frames,
            ik_anchor_frames_control=(anchor_frames // substeps).astype(np.int64),
            ik_anchor_qpos=anchor_solution_qpos,
            ik_anchor_fingertips=anchor_solution_tips,
            ik_anchor_metrics=anchor_metrics,
            waypoint_hand_joints=waypoint_hand_joints_out,
            planned_hand_joints=planned,
            planned_hand_velocities=planned_velocities.astype(np.float32),
            planned_hand_joints_dense=planned_dense.astype(np.float32),
            planned_hand_velocities_dense=dense_velocities.astype(np.float32),
            segment_ids=np.asarray(bagatelle_plan.segment_ids, dtype=np.int32).copy(),
            segment_ids_dense=segment_ids_dense.astype(np.int32),
            metadata=metadata,
        )
    finally:
        if owns_kinematics:
            kin.close()
