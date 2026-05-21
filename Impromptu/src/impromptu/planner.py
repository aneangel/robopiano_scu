from __future__ import annotations

from typing import Any

import numpy as np

from impromptu.config import ImpromptuConfig
from impromptu.anchor_selection import select_ik_anchor_frames
from impromptu.fingertip_trajectory import NUM_FINGERS, FingertipTrajectory, build_fingertip_trajectory
from impromptu.ik_solver import IK_ANCHOR_METRIC_COLUMNS, interpolate_anchor_qpos, solve_fingertip_trajectory_anchors
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

    Keep this intentionally narrow. Runtime selection comes from Impromptu,
    while assignment, key-target construction, IK objective weights, joint
    order, and inter-waypoint planning stay on Bagatelle defaults unless the
    caller passes an explicit Bagatelle kinematics object.
    """
    return BagatelleConfig(
        control_timestep=float(config.control_timestep),
        threshold=float(config.threshold),
        environment_name=str(config.environment_name),
        seed=int(config.seed),
        reduced_action_space=bool(config.reduced_action_space),
    )


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
    cfg = config or ImpromptuConfig()
    keys = validate_target_keys(target_keys)
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
        anchor_frames = select_ik_anchor_frames(
            fingertip_targets=fingertip_traj.targets,
            fingertip_weights=fingertip_traj.weights,
            waypoint_frames_dense=waypoint_frames_dense,
            config=cfg,
        )
        anchor_seed_qpos = bagatelle_dense_seed[anchor_frames] if anchor_frames.size else np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
        neutral_qpos = np.asarray(getattr(kin, "neutral_qpos", np.zeros((HAND_STATE_DIM,), dtype=np.float32)), dtype=np.float32)
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
                        anchor_solution_qpos[row] = ((1.0 - blend) * anchor_solution_qpos[row] + blend * seed).astype(
                            np.float32
                        )
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
        refinement_metadata: dict[str, object] = {"enabled": False, "chunks": 0, "optimizer_success_count": 0, "nfev_sum": 0}
        planned_dense, refinement_metadata = refine_dense_qpos_trajectory(
            kin=kin,
            dense_qpos=planned_dense,
            fingertip_targets=fingertip_traj.targets,
            fingertip_weights=fingertip_traj.weights,
            neutral_qpos=neutral_qpos,
            config=cfg,
        )
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
        metadata["ik_source"] = "Impromptu.solve_fingertip_trajectory_anchors"
        metadata["assignment_source"] = "Bagatelle.plan_target_keys"
        metadata["planned_hand_joints_source"] = "downsampled_from_dense_anchor_ik"
        metadata["dense_hand_joints_source"] = "interpolated_from_impromptu_dense_ik_anchors"
        metadata["dense_ik_seed_source"] = "interpolated_bagatelle_control_frame_qpos"
        metadata["selected_anchor_config"] = {
            "anchor_stride": int(cfg.anchor_stride),
            "solve_contact_window_only": bool(cfg.solve_contact_window_only),
            "include_midpoint_anchors": bool(cfg.include_midpoint_anchors),
            "anchor_change_threshold": float(cfg.anchor_change_threshold),
        }
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
            waypoint_hand_joints=bagatelle_plan.waypoint_hand_joints,
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
