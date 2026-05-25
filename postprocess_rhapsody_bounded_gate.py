#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for _path in (
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Rhapsody" / "src",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.assignment import FingerAssignmentResult  # noqa: E402
from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics, IKResult  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402


def _interp_rows(anchor_x: np.ndarray, anchor_y: np.ndarray, out_x: np.ndarray) -> np.ndarray:
    anchor_x = np.asarray(anchor_x, dtype=np.float64).reshape(-1)
    anchor_y = np.asarray(anchor_y, dtype=np.float32)
    out_x = np.asarray(out_x, dtype=np.float64).reshape(-1)
    out = np.empty((out_x.size, anchor_y.shape[1]), dtype=np.float32)
    for col in range(anchor_y.shape[1]):
        out[:, col] = np.interp(out_x, anchor_x, anchor_y[:, col]).astype(np.float32)
    return out


def _velocity(qpos: np.ndarray, dt: float) -> np.ndarray:
    if qpos.shape[0] <= 1:
        return np.zeros_like(qpos, dtype=np.float32)
    return np.gradient(qpos.astype(np.float32), float(dt), axis=0).astype(np.float32)


def _copy_payload(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {key: np.asarray(data[key]) for key in data.files}


def _result_for_row(
    *,
    dense_assignment: np.ndarray,
    dense_targets: np.ndarray,
) -> FingerAssignmentResult:
    active_fingers = np.flatnonzero(np.asarray(dense_assignment, dtype=np.int32).reshape(10) >= 0).astype(np.int32)
    assigned_keys = np.asarray(dense_assignment, dtype=np.int32).reshape(10)[active_fingers].astype(np.int32)
    targets = np.asarray(dense_targets, dtype=np.float32).reshape(10, 3)[active_fingers].astype(np.float32)
    key_positions = np.arange(active_fingers.size, dtype=np.int32)
    cost_matrix = np.zeros((10, max(active_fingers.size, 1)), dtype=np.float32)
    return FingerAssignmentResult(
        active_keys=assigned_keys.copy(),
        assigned_finger_indices=active_fingers,
        assigned_keys=assigned_keys,
        assigned_key_positions=key_positions,
        target_positions=targets,
        unassigned_keys=np.zeros((0,), dtype=np.int32),
        cost_matrix=cost_matrix,
        total_cost=0.0,
        mean_cost=0.0,
        strategy="rhapsody_bounded_existing_assignment",
    )


def _load_source_config(source: Path, args: argparse.Namespace) -> BagatelleConfig:
    metadata_path = source.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    source_cfg = metadata.get("bagatelle_config", metadata.get("config", {}))
    valid = {field.name for field in fields(BagatelleConfig)}
    kwargs = {key: value for key, value in source_cfg.items() if key in valid}
    kwargs.update(
        rhapsody_ik_enabled=True,
        rhapsody_ik_checkpoint=str(args.checkpoint),
        rhapsody_ik_refinement_steps=int(args.refinement_steps),
        rhapsody_ik_refinement_lr=float(args.refinement_lr),
        rhapsody_ik_device=str(args.device),
        rhapsody_ik_candidate_scoring=False,
        rhapsody_ik_coordinate_transform=str(args.coordinate_transform),
        rhapsody_ik_y_offset=float(args.y_offset),
        rhapsody_ik_fill_inactive_from_previous=True,
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
        environment_name=str(args.environment_name),
        seed=int(args.seed),
    )
    return BagatelleConfig(**kwargs)


def _null_result(assignments: FingerAssignmentResult, previous: np.ndarray, kin: BagatelleKinematics) -> IKResult:
    tips = kin.fingertip_positions_for_qpos(previous)
    return IKResult(
        pose=previous.astype(np.float32),
        fingertip_positions=tips.astype(np.float32),
        assigned_distances=np.zeros((0,), dtype=np.float32),
        residual_norm=0.0,
        max_residual=0.0,
        success=True,
        optimizer_success=True,
        optimizer_status=0,
        optimizer_message="no active assignment",
        optimizer_cost=0.0,
        nfev=0,
        active_keys=assignments.active_keys,
        assigned_keys=assignments.assigned_keys,
        assigned_finger_indices=assignments.assigned_finger_indices,
        unassigned_keys=assignments.unassigned_keys,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(source, allow_pickle=False) as data:
        payload = _copy_payload(data)

    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)
    waypoint_frames = np.asarray(payload["waypoint_frames"], dtype=np.int64).reshape(-1)
    sparse_targets = np.asarray(payload["fingertip_targets"], dtype=np.float32)
    assignments_dense = np.asarray(payload["assignments"], dtype=np.int32)
    if assignments_dense.shape != (waypoint_frames.size, 10):
        raise ValueError(f"Unexpected assignments shape {assignments_dense.shape}")

    config = _load_source_config(source, args)
    waypoint_qpos = np.asarray(payload["waypoint_hand_joints"], dtype=np.float32).copy()
    waypoint_tips = np.asarray(payload["waypoint_fingertips"], dtype=np.float32).copy()
    metrics = np.zeros((waypoint_frames.size, 7), dtype=np.float32)
    errors: list[np.ndarray] = []
    successes = 0
    optimizer_successes = 0
    failed = 0

    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=output_dir / "kinematics") as kin:
        previous = kin.neutral_qpos.astype(np.float32)
        for idx in range(waypoint_frames.size):
            assignment = _result_for_row(dense_assignment=assignments_dense[idx], dense_targets=sparse_targets[idx])
            if assignment.count == 0:
                result = _null_result(assignment, previous, kin)
            else:
                try:
                    result = kin.solve_press_pose(assignment, previous, config=config)
                except Exception as exc:
                    print(f"waypoint {idx} failed: {exc}", file=sys.stderr)
                    failed += 1
                    result = _null_result(assignment, previous, kin)
            waypoint_qpos[idx] = result.pose
            waypoint_tips[idx] = result.fingertip_positions
            previous = result.pose.astype(np.float32)
            if result.success:
                successes += 1
            if result.optimizer_success:
                optimizer_successes += 1
            if assignment.count:
                active = assignment.assigned_finger_indices.astype(np.int64)
                row_errors = np.linalg.norm(result.fingertip_positions[active] - assignment.target_positions, axis=1)
                errors.append(row_errors.astype(np.float32))
                mean_dist = float(np.mean(row_errors))
                max_dist = float(np.max(row_errors))
            else:
                mean_dist = 0.0
                max_dist = 0.0
            metrics[idx] = np.asarray(
                [
                    float(result.success),
                    float(result.optimizer_success),
                    float(result.nfev),
                    float(result.optimizer_cost),
                    mean_dist,
                    max_dist,
                    float(result.residual_norm),
                ],
                dtype=np.float32,
            )

    control_steps = int(np.asarray(payload["planned_hand_joints"]).shape[0])
    substeps = int(np.asarray(payload["planned_hand_joints_dense"]).shape[0] // max(control_steps, 1))
    control_x = np.arange(control_steps, dtype=np.float64)
    dense_x = np.arange(control_steps * substeps, dtype=np.float64) / float(substeps)
    control_qpos = _interp_rows(waypoint_frames, waypoint_qpos, control_x)
    dense_qpos = _interp_rows(control_x, control_qpos, dense_x)
    dense_dt = float(args.control_timestep) / float(substeps)

    payload["waypoint_hand_joints"] = waypoint_qpos.astype(np.float32)
    payload["waypoint_fingertips"] = waypoint_tips.astype(np.float32)
    payload["planned_hand_joints"] = control_qpos.astype(np.float32)
    payload["planned_hand_joints_dense"] = dense_qpos.astype(np.float32)
    payload["planned_hand_velocities"] = _velocity(control_qpos, float(args.control_timestep))
    payload["planned_hand_velocities_dense"] = _velocity(dense_qpos, dense_dt)
    payload["ik_anchor_qpos"] = _interp_rows(control_x, control_qpos, np.asarray(payload["ik_anchor_frames_dense"], dtype=np.float64) / float(substeps)).astype(np.float32)
    payload["ik_anchor_metrics"] = np.resize(metrics, np.asarray(payload["ik_anchor_metrics"]).shape).astype(np.float32)

    trajectory_path = output_dir / "trajectory.npz"
    atomic_save_npz(trajectory_path, **payload)
    if errors:
        all_errors = np.concatenate(errors, axis=0)
        mean_error = float(np.mean(all_errors))
        p90_error = float(np.quantile(all_errors, 0.9))
        max_error = float(np.max(all_errors))
    else:
        mean_error = p90_error = max_error = 0.0
    metadata = {
        "source_npz": str(source),
        "trajectory_npz": str(trajectory_path),
        "checkpoint": str(args.checkpoint),
        "waypoints": int(waypoint_frames.size),
        "failed_waypoints": int(failed),
        "success_count": int(successes),
        "optimizer_success_count": int(optimizer_successes),
        "mean_active_fingertip_error_m": mean_error,
        "p90_active_fingertip_error_m": p90_error,
        "max_active_fingertip_error_m": max_error,
        "refinement_steps": int(args.refinement_steps),
        "coordinate_transform": str(args.coordinate_transform),
        "y_offset": float(args.y_offset),
    }
    atomic_save_json(output_dir / "rhapsody_bounded_gate.json", metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-solve existing Impromptu assignments using bounded Bagatelle IK seeded by Rhapsody.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument("--refinement-lr", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coordinate-transform", choices=("bagatelle_to_rp1m", "none"), default="bagatelle_to_rp1m")
    parser.add_argument("--y-offset", type=float, default=0.08289646)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
