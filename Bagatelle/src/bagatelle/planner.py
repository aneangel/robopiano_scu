from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from bagatelle.assignment import (
    NUM_FINGERS,
    assign_fingers_previous_pose,
    assign_fingers_previous_pose_lookahead,
    assignment_crossing_penalty,
    generate_assignment_candidates,
)
from bagatelle.config import BagatelleConfig
from bagatelle.kinematics import HAND_STATE_DIM, BagatelleKinematics, IKResult
from bagatelle.paths import ensure_repo_paths

ensure_repo_paths()
from intermezzo.constants import LEFT_FOREARM_TY_INDEX, RIGHT_FOREARM_TY_INDEX  # noqa: E402
from intermezzo.keys import extract_waypoint_frames, validate_target_keys  # noqa: E402
from intermezzo.planner import PlannerConfig, compute_hand_velocities, plan_between_waypoints  # noqa: E402


IK_METRIC_COLUMNS = (
    "success",
    "optimizer_success",
    "nfev",
    "optimizer_cost",
    "mean_assigned_distance",
    "max_assigned_distance",
    "residual_norm",
)


@dataclass(frozen=True)
class BagatelleTrajectory:
    target_keys: np.ndarray
    waypoint_frames: np.ndarray
    waypoint_target_keys: np.ndarray
    waypoint_hand_joints: np.ndarray
    planned_hand_joints: np.ndarray
    planned_hand_velocities: np.ndarray
    segment_ids: np.ndarray
    assignments: np.ndarray
    assignment_costs: np.ndarray
    fingertip_targets: np.ndarray
    waypoint_fingertips: np.ndarray
    unassigned_keys: np.ndarray
    ik_metrics: np.ndarray
    metadata: dict[str, Any]

    def npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "target_keys": self.target_keys,
            "waypoint_frames": self.waypoint_frames,
            "waypoint_target_keys": self.waypoint_target_keys,
            "waypoint_hand_joints": self.waypoint_hand_joints,
            "planned_hand_joints": self.planned_hand_joints,
            "planned_hand_velocities": self.planned_hand_velocities,
            "segment_ids": self.segment_ids,
            "assignments": self.assignments,
            "assignment_costs": self.assignment_costs,
            "fingertip_targets": self.fingertip_targets,
            "waypoint_fingertips": self.waypoint_fingertips,
            "unassigned_keys": self.unassigned_keys,
            "ik_metrics": self.ik_metrics,
            "ik_metric_columns": np.asarray(IK_METRIC_COLUMNS),
        }


def _planner_config(config: BagatelleConfig) -> PlannerConfig:
    return PlannerConfig(
        control_timestep=float(config.control_timestep),
        threshold=float(config.threshold),
        clearance_height=float(config.clearance_height),
        lift_fraction=float(config.lift_fraction),
        descent_fraction=float(config.descent_fraction),
        vertical_min=float(config.vertical_min),
        vertical_max=float(config.vertical_max),
    )


def _empty_unassigned(rows: list[np.ndarray]) -> np.ndarray:
    width = max((int(row.size) for row in rows), default=0)
    out = np.full((len(rows), width), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        if row.size:
            out[index, : row.size] = row.astype(np.int32)
    return out


def _ik_metric_row(result: IKResult) -> np.ndarray:
    mean_distance = float(np.mean(result.assigned_distances)) if result.assigned_distances.size else 0.0
    return np.asarray(
        [
            float(result.success),
            float(result.optimizer_success),
            float(result.nfev),
            float(result.optimizer_cost),
            mean_distance,
            float(result.max_residual),
            float(result.residual_norm),
        ],
        dtype=np.float32,
    )


def _metadata_from_results(
    *,
    config: BagatelleConfig,
    target_keys: np.ndarray,
    waypoint_frames: np.ndarray,
    ik_results: list[IKResult],
    assignments: list[Any],
    kinematics: Any,
) -> dict[str, Any]:
    max_residuals = np.asarray([result.max_residual for result in ik_results], dtype=np.float32)
    unassigned_total = int(sum(int(result.unassigned_keys.size) for result in ik_results))
    metadata: dict[str, Any] = {
        "planner": "bagatelle_ot_ik_previous_pose",
        "config": config.to_dict(),
        "target_keys_shape": list(target_keys.shape),
        "num_waypoints": int(waypoint_frames.size),
        "waypoint_frames": waypoint_frames.astype(int).tolist(),
        "finger_order": "left_hand_sites_then_right_hand_sites",
        "joint_order": "right_hand_joints_then_left_hand_joints",
        "right_forearm_ty_index": int(RIGHT_FOREARM_TY_INDEX),
        "left_forearm_ty_index": int(LEFT_FOREARM_TY_INDEX),
        "ik_metric_columns": list(IK_METRIC_COLUMNS),
        "ik_success_count": int(sum(bool(result.success) for result in ik_results)),
        "ik_optimizer_success_count": int(sum(bool(result.optimizer_success) for result in ik_results)),
        "ik_unassigned_key_count": unassigned_total,
        "ik_max_residual_mean": float(max_residuals.mean()) if max_residuals.size else 0.0,
        "ik_max_residual_p95": float(np.percentile(max_residuals, 95)) if max_residuals.size else 0.0,
        "environment_name": str(getattr(kinematics, "environment_name", "")),
        "midi_proto_path": str(getattr(kinematics, "midi_proto_path", "")),
        "load_info": getattr(kinematics, "load_info", {}),
        "waypoint_results": [result.to_dict() for result in ik_results[:500]],
        "assignment_strategy": str(config.assignment_strategy),
        "assignment_top_k": int(config.assignment_top_k),
        "assignment_cost_weights": {
            "distance": float(config.assignment_distance_weight),
            "hand_zone": float(config.assignment_hand_zone_weight),
            "finger_zone": float(config.assignment_finger_zone_weight),
            "crossing": float(config.assignment_crossing_weight),
            "hold": float(config.assignment_hold_weight),
            "reach": float(config.assignment_reach_weight),
            "black_key": float(config.assignment_black_key_weight),
            "ik_residual": float(config.assignment_ik_residual_weight),
            "ik_max_residual": float(config.assignment_ik_max_residual_weight),
            "motion": float(config.assignment_motion_weight),
        },
    }
    candidate_ranks = [
        int(getattr(assignment, "candidate_rank", 0))
        for assignment in assignments
        if getattr(assignment, "candidate_score", None) is not None
    ]
    candidate_scores = [
        float(getattr(assignment, "candidate_score"))
        for assignment in assignments
        if getattr(assignment, "candidate_score", None) is not None
    ]
    if candidate_ranks:
        rank_array = np.asarray(candidate_ranks, dtype=np.float32)
        metadata["assignment_candidate_rank_mean"] = float(np.mean(rank_array))
        metadata["assignment_candidate_rank_max"] = int(np.max(rank_array))
    if candidate_scores:
        score_array = np.asarray(candidate_scores, dtype=np.float32)
        metadata["assignment_candidate_score_mean"] = float(np.mean(score_array))
    return metadata


def plan_target_keys(
    target_keys: np.ndarray,
    config: BagatelleConfig | None = None,
    *,
    kinematics: Any | None = None,
) -> BagatelleTrajectory:
    cfg = config or BagatelleConfig()
    keys = validate_target_keys(target_keys)
    waypoint_frames = extract_waypoint_frames(keys, threshold=float(cfg.threshold))
    waypoint_target_keys = keys[waypoint_frames] if waypoint_frames.size else np.zeros((0, 88), dtype=np.float32)

    owns_kinematics = kinematics is None
    kin = kinematics or BagatelleKinematics(cfg, target_keys=keys)
    try:
        previous_qpos = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
        neutral_qpos = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
        previous_fingertips = np.asarray(kin.fingertip_positions_for_qpos(previous_qpos), dtype=np.float32)
        previous_dense_assignment = np.full((NUM_FINGERS,), -1, dtype=np.int32)

        active_key_rows = [
            np.flatnonzero(row[:88] > float(cfg.threshold)).astype(np.int32) for row in waypoint_target_keys
        ]
        contact_target_rows = [kin.key_contact_targets(active_keys) for active_keys in active_key_rows]
        sequence_cost_biases: np.ndarray | None = None
        if str(cfg.sequence_model_checkpoint) and str(cfg.sequence_model_type).lower() != "none":
            from bagatelle.sequence_model import CostBiasAssigner

            contact_targets_all = kin.key_contact_targets(np.arange(88, dtype=np.int32))
            sequence_assigner = CostBiasAssigner.from_checkpoint(
                cfg.sequence_model_checkpoint,
                model_type=str(cfg.sequence_model_type),
                alpha=float(cfg.cost_bias_alpha),
            )
            sequence_cost_biases = sequence_assigner.assign_all(waypoint_target_keys, contact_targets_all)

        waypoint_poses: list[np.ndarray] = []
        waypoint_fingertips: list[np.ndarray] = []
        assignment_rows: list[np.ndarray] = []
        assignment_cost_rows: list[np.ndarray] = []
        fingertip_target_rows: list[np.ndarray] = []
        unassigned_rows: list[np.ndarray] = []
        ik_metric_rows: list[np.ndarray] = []
        ik_results: list[IKResult] = []
        final_assignments: list[Any] = []

        strategy = str(getattr(cfg, "assignment_strategy", "legacy_previous_pose"))

        for waypoint_index, _target_row in enumerate(waypoint_target_keys):
            active_keys = active_key_rows[waypoint_index]
            contact_targets = contact_target_rows[waypoint_index]
            cost_bias = None if sequence_cost_biases is None else sequence_cost_biases[waypoint_index]

            if strategy == "ik_aware_topk":
                candidates = generate_assignment_candidates(
                    active_keys,
                    previous_fingertips,
                    contact_targets,
                    cfg,
                    max_candidates=int(cfg.assignment_top_k),
                    previous_assignment=previous_dense_assignment,
                )
                if not candidates:
                    assignment = assign_fingers_previous_pose(
                        active_keys,
                        previous_fingertips,
                        contact_targets,
                        cfg,
                        previous_assignment=previous_dense_assignment,
                    )
                    candidates = [replace_candidate_for_fallback(assignment)]
                press_targets = kin.key_press_targets(active_keys)
                best_assignment = None
                best_result = None
                best_score = float("inf")
                for candidate in candidates:
                    assignment = candidate.result
                    if assignment.count:
                        assignment = replace(
                            assignment,
                            target_positions=press_targets[assignment.assigned_key_positions].astype(np.float32),
                        )
                    result = kin.solve_press_pose(
                        assignment,
                        previous_qpos,
                        neutral_qpos=neutral_qpos,
                        config=cfg,
                    )
                    motion = np.linalg.norm(result.pose.astype(np.float32) - previous_qpos.astype(np.float32))
                    crossing = assignment_crossing_penalty(
                        assignment.assigned_finger_indices,
                        assignment.assigned_keys,
                        cfg,
                    )
                    score = (
                        float(candidate.base_cost)
                        + float(cfg.assignment_crossing_weight) * float(crossing)
                        + float(cfg.assignment_ik_residual_weight) * float(result.residual_norm)
                        + float(cfg.assignment_ik_max_residual_weight) * float(result.max_residual)
                        + float(cfg.assignment_motion_weight) * float(motion)
                        + (0.0 if result.success else float(cfg.assignment_ik_failure_penalty))
                    )
                    if score < best_score:
                        best_score = float(score)
                        best_assignment = replace(assignment, candidate_rank=int(candidate.rank), candidate_score=float(score))
                        best_result = result
                if best_assignment is None or best_result is None:
                    raise RuntimeError("generate_assignment_candidates returned no usable candidate")
                assignment = best_assignment
                result = best_result
            else:
                if cost_bias is not None:
                    assignment = assign_fingers_previous_pose(
                        active_keys,
                        previous_fingertips,
                        contact_targets,
                        cfg,
                        previous_assignment=previous_dense_assignment,
                        cost_bias=cost_bias,
                        cost_bias_alpha=float(cfg.cost_bias_alpha),
                    )
                elif int(cfg.assignment_lookahead_steps) > 0:
                    assignment = assign_fingers_previous_pose_lookahead(
                        active_keys,
                        previous_fingertips,
                        contact_targets,
                        future_active_keys=active_key_rows[waypoint_index + 1 :],
                        future_key_targets=contact_target_rows[waypoint_index + 1 :],
                        previous_assignment=previous_dense_assignment,
                        lookahead_steps=int(cfg.assignment_lookahead_steps),
                        config=cfg,
                        lookahead_weight=float(cfg.assignment_lookahead_weight),
                    )
                else:
                    assignment = assign_fingers_previous_pose(
                        active_keys,
                        previous_fingertips,
                        contact_targets,
                        cfg,
                        previous_assignment=previous_dense_assignment,
                    )
                press_targets = kin.key_press_targets(active_keys)
                if assignment.count:
                    assignment = replace(
                        assignment,
                        target_positions=press_targets[assignment.assigned_key_positions].astype(np.float32),
                    )
                result = kin.solve_press_pose(assignment, previous_qpos, neutral_qpos=neutral_qpos, config=cfg)

            waypoint_poses.append(result.pose.astype(np.float32))
            waypoint_fingertips.append(result.fingertip_positions.astype(np.float32))
            assignment_rows.append(assignment.dense_key_by_finger())
            assignment_cost_rows.append(assignment.dense_cost_by_finger())
            fingertip_target_rows.append(assignment.dense_targets_by_finger())
            unassigned_rows.append(assignment.unassigned_keys.astype(np.int32))
            ik_metric_rows.append(_ik_metric_row(result))
            ik_results.append(result)
            final_assignments.append(assignment)

            previous_qpos = result.pose.astype(np.float32)
            previous_fingertips = result.fingertip_positions.astype(np.float32)
            previous_dense_assignment = assignment.dense_key_by_finger()

        if waypoint_poses:
            waypoint_hand_joints = np.stack(waypoint_poses, axis=0).astype(np.float32)
            fingertip_array = np.stack(waypoint_fingertips, axis=0).astype(np.float32)
            assignments = np.stack(assignment_rows, axis=0).astype(np.int32)
            assignment_costs = np.stack(assignment_cost_rows, axis=0).astype(np.float32)
            fingertip_targets = np.stack(fingertip_target_rows, axis=0).astype(np.float32)
            ik_metrics = np.stack(ik_metric_rows, axis=0).astype(np.float32)
            unassigned = _empty_unassigned(unassigned_rows)
            planned, velocities, segment_ids, sanitized = plan_between_waypoints(
                total_steps=int(keys.shape[0]),
                waypoint_frames=waypoint_frames,
                waypoint_target_keys=waypoint_target_keys,
                waypoint_hand_joints=waypoint_hand_joints,
                config=_planner_config(cfg),
            )
            waypoint_hand_joints = sanitized.astype(np.float32)
        else:
            waypoint_hand_joints = np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
            assignments = np.zeros((0, NUM_FINGERS), dtype=np.int32)
            assignment_costs = np.zeros((0, NUM_FINGERS), dtype=np.float32)
            fingertip_targets = np.zeros((0, NUM_FINGERS, 3), dtype=np.float32)
            fingertip_array = np.zeros((0, NUM_FINGERS, 3), dtype=np.float32)
            unassigned = np.zeros((0, 0), dtype=np.int32)
            ik_metrics = np.zeros((0, len(IK_METRIC_COLUMNS)), dtype=np.float32)
            planned = np.tile(neutral_qpos.reshape(1, -1), (keys.shape[0], 1)).astype(np.float32)
            velocities = compute_hand_velocities(planned, control_timestep=float(cfg.control_timestep))
            segment_ids = np.full((keys.shape[0],), -1, dtype=np.int32)

        metadata = _metadata_from_results(
            config=cfg,
            target_keys=keys,
            waypoint_frames=waypoint_frames,
            ik_results=ik_results,
            assignments=final_assignments,
            kinematics=kin,
        )
        metadata["assignment_mode"] = (
            f"sequence_{cfg.sequence_model_type}"
            if sequence_cost_biases is not None
            else (
                f"lookahead_{int(cfg.assignment_lookahead_steps)}"
                if int(cfg.assignment_lookahead_steps) > 0 and strategy != "ik_aware_topk"
                else strategy
            )
        )
        metadata["planned_hand_joints_shape"] = list(planned.shape)
        metadata["planned_hand_velocities_shape"] = list(velocities.shape)

        return BagatelleTrajectory(
            target_keys=keys,
            waypoint_frames=waypoint_frames,
            waypoint_target_keys=waypoint_target_keys,
            waypoint_hand_joints=waypoint_hand_joints,
            planned_hand_joints=planned.astype(np.float32),
            planned_hand_velocities=velocities.astype(np.float32),
            segment_ids=segment_ids.astype(np.int32),
            assignments=assignments,
            assignment_costs=assignment_costs,
            fingertip_targets=fingertip_targets,
            waypoint_fingertips=fingertip_array,
            unassigned_keys=unassigned,
            ik_metrics=ik_metrics,
            metadata=metadata,
        )
    finally:
        if owns_kinematics:
            kin.close()


def replace_candidate_for_fallback(assignment: Any) -> Any:
    class _FallbackCandidate:
        def __init__(self, result: Any) -> None:
            self.result = result
            self.base_cost = float(result.total_cost)
            self.rank = 0

    return _FallbackCandidate(assignment)
