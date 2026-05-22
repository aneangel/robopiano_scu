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
from bagatelle.kinematics import FINGER_ORDER, HAND_STATE_DIM, JOINT_ORDER, JOINT_INDEX_RANGES_BY_HAND, BagatelleKinematics, IKResult
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


@dataclass(frozen=True)
class _SequenceBeamPlanState:
    qpos: np.ndarray
    fingertips: np.ndarray
    dense_assignment: np.ndarray
    cumulative_cost: float
    waypoint_poses: tuple[np.ndarray, ...]
    waypoint_fingertips: tuple[np.ndarray, ...]
    assignment_rows: tuple[np.ndarray, ...]
    assignment_cost_rows: tuple[np.ndarray, ...]
    fingertip_target_rows: tuple[np.ndarray, ...]
    unassigned_rows: tuple[np.ndarray, ...]
    ik_metric_rows: tuple[np.ndarray, ...]
    ik_results: tuple[IKResult, ...]
    assignments: tuple[Any, ...]


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


def _unassigned_score(unassigned_count: int, config: BagatelleConfig) -> float:
    if int(unassigned_count) <= 0:
        return 0.0
    if bool(config.assignment_fail_if_unassigned):
        return float("inf")
    return float(config.assignment_unassigned_penalty) * float(unassigned_count)


def _assignment_candidate_diagnostic(
    *,
    waypoint_index: int,
    candidate_rank: int,
    assignment: Any,
    assignment_cost: float,
    crossing: float,
    ik_result: IKResult,
    motion: float,
    final_score: float,
    selected: bool,
) -> dict[str, Any]:
    return {
        "waypoint_index": int(waypoint_index),
        "candidate_rank": int(candidate_rank),
        "assignment_signature": assignment.dense_key_by_finger().astype(int).tolist(),
        "assignment_cost": float(assignment_cost),
        "crossing_penalty": float(crossing),
        "ik_residual": float(ik_result.residual_norm),
        "ik_max_residual": float(ik_result.max_residual),
        "ik_success": bool(ik_result.success),
        "motion_cost": float(motion),
        "final_score": float(final_score),
        "selected": bool(selected),
        "unassigned_keys": assignment.unassigned_keys.astype(int).tolist(),
    }


def _sequence_beam_score(
    *,
    candidate_base_cost: float,
    assignment: Any,
    ik_result: IKResult,
    previous_qpos: np.ndarray,
    config: BagatelleConfig,
) -> tuple[float, float, float]:
    motion = float(np.linalg.norm(ik_result.pose.astype(np.float32) - previous_qpos.astype(np.float32)))
    crossing = assignment_crossing_penalty(
        assignment.assigned_finger_indices,
        assignment.assigned_keys,
        config,
    )
    score = (
        float(candidate_base_cost)
        + float(config.assignment_crossing_weight) * float(crossing)
        + float(config.assignment_ik_residual_weight) * float(ik_result.residual_norm)
        + float(config.assignment_ik_max_residual_weight) * float(ik_result.max_residual)
        + float(config.assignment_motion_weight) * float(motion)
        + _unassigned_score(int(assignment.unassigned_keys.size), config)
        + (0.0 if ik_result.success else float(config.assignment_ik_failure_penalty))
    )
    return float(score), float(crossing), float(motion)


def _plan_sequence_beam(
    *,
    cfg: BagatelleConfig,
    kin: Any,
    active_key_rows: list[np.ndarray],
    contact_target_rows: list[np.ndarray],
    neutral_qpos: np.ndarray,
    previous_qpos: np.ndarray,
    previous_fingertips: np.ndarray,
    previous_dense_assignment: np.ndarray,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[IKResult],
    list[Any],
    list[dict[str, Any]],
]:
    beam_width = max(int(cfg.assignment_beam_width), 1)
    candidates_per_step = int(cfg.assignment_candidates_per_step or cfg.assignment_top_k)
    candidates_per_step = max(candidates_per_step, 1)
    press_target_rows = [kin.key_press_targets(active_keys) for active_keys in active_key_rows]
    diagnostics: list[dict[str, Any]] = []
    ik_cache: dict[tuple[int, bytes, tuple[int, ...]], IKResult] = {}
    beam = [
        _SequenceBeamPlanState(
            qpos=previous_qpos.astype(np.float32),
            fingertips=previous_fingertips.astype(np.float32),
            dense_assignment=previous_dense_assignment.astype(np.int32),
            cumulative_cost=0.0,
            waypoint_poses=(),
            waypoint_fingertips=(),
            assignment_rows=(),
            assignment_cost_rows=(),
            fingertip_target_rows=(),
            unassigned_rows=(),
            ik_metric_rows=(),
            ik_results=(),
            assignments=(),
        )
    ]

    for waypoint_index, active_keys in enumerate(active_key_rows):
        expansions: list[tuple[float, int, _SequenceBeamPlanState, dict[str, Any]]] = []
        for state_index, state in enumerate(beam):
            candidates = generate_assignment_candidates(
                active_keys,
                state.fingertips,
                contact_target_rows[waypoint_index],
                cfg,
                max_candidates=candidates_per_step,
                previous_assignment=state.dense_assignment,
            )
            for candidate in candidates:
                assignment = candidate.result
                if assignment.count:
                    assignment = replace(
                        assignment,
                        target_positions=press_target_rows[waypoint_index][
                            assignment.assigned_key_positions
                        ].astype(np.float32),
                    )
                cache_key = (
                    waypoint_index,
                    np.round(state.qpos.astype(np.float32), 5).tobytes(),
                    tuple(assignment.dense_key_by_finger().astype(int).tolist()),
                )
                result = ik_cache.get(cache_key)
                if result is None:
                    result = kin.solve_press_pose(
                        assignment,
                        state.qpos,
                        neutral_qpos=neutral_qpos,
                        config=cfg,
                    )
                    ik_cache[cache_key] = result
                local_score, crossing, motion = _sequence_beam_score(
                    candidate_base_cost=float(candidate.base_cost),
                    assignment=assignment,
                    ik_result=result,
                    previous_qpos=state.qpos,
                    config=cfg,
                )
                if not np.isfinite(local_score):
                    continue
                cumulative = float(state.cumulative_cost + local_score)
                scored_assignment = replace(
                    assignment,
                    strategy="sequence_beam",
                    candidate_rank=int(candidate.rank),
                    candidate_score=cumulative,
                )
                diagnostic = _assignment_candidate_diagnostic(
                    waypoint_index=waypoint_index,
                    candidate_rank=int(candidate.rank),
                    assignment=scored_assignment,
                    assignment_cost=float(candidate.base_cost),
                    crossing=float(crossing),
                    ik_result=result,
                    motion=float(motion),
                    final_score=cumulative,
                    selected=False,
                )
                next_state = _SequenceBeamPlanState(
                    qpos=result.pose.astype(np.float32),
                    fingertips=result.fingertip_positions.astype(np.float32),
                    dense_assignment=scored_assignment.dense_key_by_finger(),
                    cumulative_cost=cumulative,
                    waypoint_poses=state.waypoint_poses + (result.pose.astype(np.float32),),
                    waypoint_fingertips=state.waypoint_fingertips + (result.fingertip_positions.astype(np.float32),),
                    assignment_rows=state.assignment_rows + (scored_assignment.dense_key_by_finger(),),
                    assignment_cost_rows=state.assignment_cost_rows + (scored_assignment.dense_cost_by_finger(),),
                    fingertip_target_rows=state.fingertip_target_rows + (scored_assignment.dense_targets_by_finger(),),
                    unassigned_rows=state.unassigned_rows + (scored_assignment.unassigned_keys.astype(np.int32),),
                    ik_metric_rows=state.ik_metric_rows + (_ik_metric_row(result),),
                    ik_results=state.ik_results + (result,),
                    assignments=state.assignments + (scored_assignment,),
                )
                expansions.append((cumulative, state_index, next_state, diagnostic))
        if not expansions:
            raise RuntimeError("sequence_beam found no usable candidate path")
        expansions.sort(
            key=lambda item: (
                float(item[0]),
                int(item[1]),
                tuple(item[2].dense_assignment.astype(int).tolist()),
            )
        )
        selected_ids = {id(item[2]) for item in expansions[:beam_width]}
        if waypoint_index < 500:
            for _, _, state, diagnostic in expansions:
                diagnostics.append({**diagnostic, "selected": id(state) in selected_ids})
        beam = [state for _, _, state, _ in expansions[:beam_width]]

    best = min(beam, key=lambda state: (float(state.cumulative_cost), tuple(state.dense_assignment.astype(int).tolist())))
    return (
        list(best.waypoint_poses),
        list(best.waypoint_fingertips),
        list(best.assignment_rows),
        list(best.assignment_cost_rows),
        list(best.fingertip_target_rows),
        list(best.unassigned_rows),
        list(best.ik_metric_rows),
        list(best.ik_results),
        list(best.assignments),
        diagnostics,
    )


def _metadata_from_results(
    *,
    config: BagatelleConfig,
    target_keys: np.ndarray,
    waypoint_frames: np.ndarray,
    ik_results: list[IKResult],
    assignments: list[Any],
    kinematics: Any,
    candidate_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    max_residuals = np.asarray([result.max_residual for result in ik_results], dtype=np.float32)
    unassigned_total = int(sum(int(result.unassigned_keys.size) for result in ik_results))
    site_names = list(getattr(kinematics, "fingertip_site_names", ()))
    joint_names = list(getattr(kinematics, "joint_names", ()))
    order_metadata = getattr(kinematics, "order_metadata", {}) or {}
    metadata: dict[str, Any] = {
        "planner": "bagatelle_ot_ik_previous_pose",
        "config": config.to_dict(),
        "target_keys_shape": list(target_keys.shape),
        "num_waypoints": int(waypoint_frames.size),
        "waypoint_frames": waypoint_frames.astype(int).tolist(),
        "finger_order": str(order_metadata.get("finger_order", FINGER_ORDER)),
        "joint_order": str(order_metadata.get("joint_order", JOINT_ORDER)),
        "fingertip_site_names": list(order_metadata.get("fingertip_site_names", site_names)),
        "joint_names": list(order_metadata.get("joint_names", joint_names)),
        "finger_hands": list(order_metadata.get("finger_hands", [])),
        "finger_names": list(order_metadata.get("finger_names", [])),
        "assignment_finger_to_site": list(order_metadata.get("assignment_finger_to_site", [
            {"finger_index": index, "site_name": name}
            for index, name in enumerate(site_names)
        ])),
        "joint_index_ranges_by_hand": dict(order_metadata.get("joint_index_ranges_by_hand", JOINT_INDEX_RANGES_BY_HAND)),
        "key_target_mode": "press",
        "key_target_parameters": {
            "front_offset": float(config.key_target_front_offset),
            "top_offset": float(config.key_target_top_offset),
            "press_depth": float(config.key_press_depth),
        },
        "ik_solver": {
            "max_nfev": int(config.ik_max_nfev),
            "ftol": float(config.ik_ftol),
            "xtol": float(config.ik_xtol),
            "gtol": float(config.ik_gtol),
            "residual_success_threshold": float(config.residual_success_threshold),
            "weights": {
                "fingertip": float(config.ik_fingertip_weight),
                "key_front": float(config.ik_key_front_weight),
                "key_width": float(config.ik_key_width_weight),
                "key_height": float(config.ik_key_height_weight),
                "smoothness": float(config.ik_smoothness_weight),
                "neutral": float(config.ik_neutral_weight),
                "inactive_fingertip_clearance": float(config.ik_inactive_fingertip_clearance_weight),
                "unassigned_fingertip_strategy": str(config.ik_unassigned_fingertip_strategy),
                "unassigned_fingertip_avoidance": float(config.ik_unassigned_fingertip_avoidance_weight),
                "wrong_key_xy_avoidance": float(config.ik_wrong_key_xy_avoidance_weight),
            },
        },
        "per_waypoint_residual_histograms": [
            np.histogram(result.assigned_distances, bins=[0.0, 0.005, 0.01, 0.02, 0.05, np.inf])[0].astype(int).tolist()
            for result in ik_results[:500]
        ],
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
        "assignment_beam_width": int(config.assignment_beam_width),
        "assignment_candidates_per_step": int(config.assignment_candidates_per_step or config.assignment_top_k),
        "assignment_candidate_diagnostics_schema": {
            "waypoint_index": "0-based waypoint index",
            "assignment_signature": "dense key-by-finger vector, -1 for idle fingers",
            "assignment_cost": "candidate linear assignment cost before IK terms",
            "ik_residual": "L2 norm of assigned fingertip residuals",
            "motion_cost": "L2 qpos motion from prior beam state",
            "final_score": "cumulative beam score after this candidate",
            "selected": "candidate survived this waypoint's beam pruning",
        },
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
    if candidate_diagnostics is not None:
        metadata["assignment_candidate_diagnostics"] = candidate_diagnostics[:500]
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
        candidate_diagnostics: list[dict[str, Any]] = []

        strategy = str(getattr(cfg, "assignment_strategy", "legacy_previous_pose"))

        if strategy == "sequence_beam":
            (
                waypoint_poses,
                waypoint_fingertips,
                assignment_rows,
                assignment_cost_rows,
                fingertip_target_rows,
                unassigned_rows,
                ik_metric_rows,
                ik_results,
                final_assignments,
                candidate_diagnostics,
            ) = _plan_sequence_beam(
                cfg=cfg,
                kin=kin,
                active_key_rows=active_key_rows,
                contact_target_rows=contact_target_rows,
                neutral_qpos=neutral_qpos,
                previous_qpos=previous_qpos,
                previous_fingertips=previous_fingertips,
                previous_dense_assignment=previous_dense_assignment,
            )

        for waypoint_index, _target_row in enumerate([] if strategy == "sequence_beam" else waypoint_target_keys):
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
                    score, crossing, motion = _sequence_beam_score(
                        candidate_base_cost=float(candidate.base_cost),
                        assignment=assignment,
                        ik_result=result,
                        previous_qpos=previous_qpos,
                        config=cfg,
                    )
                    if waypoint_index < 500:
                        candidate_diagnostics.append(
                            _assignment_candidate_diagnostic(
                                waypoint_index=waypoint_index,
                                candidate_rank=int(candidate.rank),
                                assignment=assignment,
                                assignment_cost=float(candidate.base_cost),
                                crossing=float(crossing),
                                ik_result=result,
                                motion=float(motion),
                                final_score=float(score),
                                selected=False,
                            )
                        )
                    if score < best_score:
                        best_score = float(score)
                        best_assignment = replace(assignment, candidate_rank=int(candidate.rank), candidate_score=float(score))
                        best_result = result
                if best_assignment is None or best_result is None:
                    raise RuntimeError("generate_assignment_candidates returned no usable candidate")
                assignment = best_assignment
                result = best_result
                if waypoint_index < 500 and candidate_diagnostics:
                    for diagnostic in reversed(candidate_diagnostics):
                        if diagnostic["waypoint_index"] != waypoint_index:
                            break
                        diagnostic["selected"] = (
                            int(diagnostic["candidate_rank"]) == int(assignment.candidate_rank)
                            and diagnostic["assignment_signature"] == assignment.dense_key_by_finger().astype(int).tolist()
                        )
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
            candidate_diagnostics=candidate_diagnostics,
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
