#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402


def _interp_rows(anchor_x: np.ndarray, anchor_y: np.ndarray, out_x: np.ndarray) -> np.ndarray:
    anchor_x = np.asarray(anchor_x, dtype=np.float64).reshape(-1)
    anchor_y = np.asarray(anchor_y, dtype=np.float32)
    out_x = np.asarray(out_x, dtype=np.float64).reshape(-1)
    out = np.empty((out_x.size, anchor_y.shape[1]), dtype=np.float32)
    for col in range(anchor_y.shape[1]):
        out[:, col] = np.interp(out_x, anchor_x, anchor_y[:, col]).astype(np.float32)
    return out


def _velocity(control_qpos: np.ndarray, dt: float) -> np.ndarray:
    if control_qpos.shape[0] <= 1:
        return np.zeros_like(control_qpos, dtype=np.float32)
    return np.gradient(control_qpos.astype(np.float32), float(dt), axis=0).astype(np.float32)


def _copy_payload(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {key: np.asarray(data[key]) for key in data.files}


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.npz"
    metadata_path = output_dir / "rhapsody_dense_gate.json"

    with np.load(source, allow_pickle=False) as data:
        payload = _copy_payload(data)

    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)
    waypoint_frames = np.asarray(payload["waypoint_frames"], dtype=np.int64).reshape(-1)
    sparse_targets = np.asarray(payload["fingertip_targets"], dtype=np.float32)
    if sparse_targets.shape[:2] != (waypoint_frames.size, 10):
        raise ValueError(f"Unexpected fingertip_targets shape {sparse_targets.shape}")

    config = BagatelleConfig(
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
        environment_name=str(args.environment_name),
        seed=int(args.seed),
        rhapsody_ik_enabled=True,
        rhapsody_ik_checkpoint=str(args.checkpoint),
        rhapsody_ik_refinement_steps=int(args.refinement_steps),
        rhapsody_ik_refinement_lr=float(args.refinement_lr),
        rhapsody_ik_device=str(args.device),
        rhapsody_ik_candidate_scoring=False,
        rhapsody_ik_coordinate_transform=str(args.coordinate_transform),
        rhapsody_ik_y_offset=float(args.y_offset),
        rhapsody_ik_fill_inactive_from_previous=True,
    )

    waypoint_qpos = np.asarray(payload["waypoint_hand_joints"], dtype=np.float32).copy()
    waypoint_tips = np.asarray(payload["waypoint_fingertips"], dtype=np.float32).copy()
    active_errors: list[np.ndarray] = []
    changed = 0
    failed = 0

    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=output_dir / "kinematics") as kin:
        previous = kin.neutral_qpos.astype(np.float32)
        for idx in range(waypoint_frames.size):
            targets = sparse_targets[idx]
            active = np.isfinite(targets).all(axis=1).astype(np.float32)
            if not bool(np.any(active > 0.0)):
                waypoint_qpos[idx] = previous
                waypoint_tips[idx] = kin.fingertip_positions_for_qpos(previous)
                continue
            try:
                qpos = kin.rhapsody_seed_for_fingertips(targets, active, previous, config=config)
                tips = kin.fingertip_positions_for_qpos(qpos)
                waypoint_qpos[idx] = qpos
                waypoint_tips[idx] = tips
                previous = qpos
                changed += 1
                active_idx = active > 0.0
                active_errors.append(np.linalg.norm(tips[active_idx] - targets[active_idx], axis=1).astype(np.float32))
            except Exception as exc:
                failed += 1
                print(f"waypoint {idx} failed: {exc}", file=sys.stderr)
                previous = np.asarray(waypoint_qpos[idx], dtype=np.float32)

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

    if "ik_anchor_frames_dense" in payload and "ik_anchor_qpos" in payload:
        anchor_dense = np.asarray(payload["ik_anchor_frames_dense"], dtype=np.int64).reshape(-1)
        if anchor_dense.size:
            anchor_x = np.clip(anchor_dense.astype(np.float64) / float(substeps), 0.0, float(control_steps - 1))
            payload["ik_anchor_qpos"] = _interp_rows(control_x, control_qpos, anchor_x).astype(np.float32)

    atomic_save_npz(trajectory_path, **payload)
    if active_errors:
        errors = np.concatenate(active_errors, axis=0)
        mean_active_error = float(np.mean(errors))
        max_active_error = float(np.max(errors))
        p90_active_error = float(np.quantile(errors, 0.9))
    else:
        mean_active_error = max_active_error = p90_active_error = 0.0
    metadata = {
        "source_npz": str(source),
        "trajectory_npz": str(trajectory_path),
        "checkpoint": str(args.checkpoint),
        "changed_waypoints": int(changed),
        "failed_waypoints": int(failed),
        "waypoints": int(waypoint_frames.size),
        "control_steps": int(control_steps),
        "interpolation_substeps": int(substeps),
        "refinement_steps": int(args.refinement_steps),
        "coordinate_transform": str(args.coordinate_transform),
        "y_offset": float(args.y_offset),
        "mean_active_fingertip_error_m": mean_active_error,
        "p90_active_fingertip_error_m": p90_active_error,
        "max_active_fingertip_error_m": max_active_error,
    }
    atomic_save_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate Rhapsody dense inactive-finger IK on an existing Impromptu trajectory.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument("--refinement-lr", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coordinate-transform", choices=("bagatelle_to_rp1m", "none"), default="bagatelle_to_rp1m")
    parser.add_argument("--y-offset", type=float, default=0.08289646)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
