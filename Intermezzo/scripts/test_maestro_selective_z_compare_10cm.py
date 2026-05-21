from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path("/WAVE/projects/ECEN-524-Wi26/robopiano")
for path in [REPO / "Bagatelle/src", REPO / "Intermezzo/src", REPO / "partita/src", REPO / "Variations/src", REPO / "Variations", REPO]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402
from bagatelle.planner import plan_target_keys  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir  # noqa: E402
from intermezzo.constants import LEFT_FOREARM_TY_INDEX, RIGHT_FOREARM_TY_INDEX  # noqa: E402
from intermezzo.keys import keyset_hand_sides  # noqa: E402
from intermezzo.magnetic_field import frame_window_envelope  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402
from intermezzo.online_eval import RolloutConfig, score_rollout  # noqa: E402
from intermezzo.planner import PlannerConfig, plan_between_waypoints  # noqa: E402
from partita.evaluation.rollout import (  # noqa: E402
    _capture_piano_activation,
    _load_env,
    _locate_task_physics_piano,
    _set_reduced_hand_qpos,
    candidate_environment_names,
    piano_roll_to_midi_events,
    render_frame,
    write_goals_proto,
    write_video,
)


MAESTRO_MIDI = Path(
    "/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0/"
    "2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.midi"
)
OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Intermezzo/rendered_online")
ANISOTROPIC_RADII_XYZ = np.asarray([0.10, 0.10, 0.10], dtype=np.float32)


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    condition_label = os.environ.get("MAESTRO_CONDITION_LABEL", "below_key").strip() or "below_key"
    key_press_depth = float(os.environ.get("BAGATELLE_KEY_PRESS_DEPTH", "0.005"))
    run_name = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}_"
        f"maestro_liszt_preludio_bagatelle_{condition_label}_selective_z_mask_10cm_decay_200hz_render"
    )
    run_dir = create_unique_run_dir(OUTPUT_ROOT, run_name=run_name)

    target_keys, quant_meta = load_target_keys_from_midi(MAESTRO_MIDI, control_timestep=0.05)
    bagatelle_config = BagatelleConfig(
        control_timestep=0.05,
        threshold=0.5,
        seed=0,
        environment_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
        ik_fingertip_weight=5.0,
        ik_key_front_weight=1.0,
        ik_key_width_weight=2.0,
        ik_key_height_weight=1.0,
        ik_smoothness_weight=0.01,
        ik_neutral_weight=0.002,
        ik_max_nfev=240,
        residual_success_threshold=0.02,
        key_press_depth=key_press_depth,
        clearance_height=0.055,
    )
    bagatelle = plan_target_keys(target_keys, config=bagatelle_config)
    bagatelle_npz = run_dir / "bagatelle_width2_target_hand_states.npz"
    atomic_save_npz(bagatelle_npz, **bagatelle.npz_payload())

    # Start from the best non-sequence Voiles force config, but disable Intermezzo's scalar XY magnet.
    planner_config = PlannerConfig(
        control_timestep=0.05,
        threshold=0.5,
        interpolation_substeps=10,
        press_approach_s=0.05,
        press_hold_s=0.03,
        press_release_s=0.05,
        press_envelope_power=1.5,
        press_depth=0.0036125,
        clearance_height=0.03,
        lift_fraction=0.20,
        descent_fraction=0.35,
        enable_key_magnetism=False,
        magnet_radius=0.06,
        magnet_sigma=0.06,
        magnet_gain=0.55,
        magnet_max_xy_step=0.0045,
        magnet_start_fraction=0.40,
        magnet_power=1.0,
        ik_damping=1e-2,
        ik_max_delta_q=0.04,
        ik_iterations_per_frame=2,
        preserve_waypoint_endpoints=True,
        magnet_only_final_keyset=True,
    )

    planned, velocities, segment_ids, sanitized, dense, dense_velocities, dense_segment_ids = plan_between_waypoints(
        total_steps=int(target_keys.shape[0]),
        waypoint_frames=bagatelle.waypoint_frames,
        waypoint_target_keys=bagatelle.waypoint_target_keys,
        waypoint_hand_joints=bagatelle.waypoint_hand_joints,
        config=planner_config,
        return_dense=True,
    )
    del planned, velocities, segment_ids, dense_velocities, dense_segment_ids

    with BagatelleKinematics(bagatelle_config, target_keys=target_keys, output_dir=run_dir / "anisotropic_magnet_kinematics") as kin:
        press_targets = kin.key_press_targets(None)
        dense_aniso, magnet_stats = apply_anisotropic_magnet(
            dense,
            waypoint_frames=bagatelle.waypoint_frames,
            waypoint_assignments=bagatelle.assignments,
            key_press_targets=press_targets,
            kinematics=kin,
            config=planner_config,
            radii_xyz=ANISOTROPIC_RADII_XYZ,
        )
        dense_aniso, z_segment_stats = apply_segment_endpoint_z_magnet(
            dense_aniso,
            waypoint_frames=bagatelle.waypoint_frames,
            waypoint_assignments=bagatelle.assignments,
            key_press_targets=press_targets,
            kinematics=kin,
            config=planner_config,
            z_radius=float(ANISOTROPIC_RADII_XYZ[2]),
            endpoint_gain=1.4,
            mid_gain=0.08,
            endpoint_width_fraction=0.16,
            z_step_limit=0.006,
        )
        dense_aniso, release_lift_stats = apply_depress_release_lift_boost(
            dense_aniso,
            waypoint_frames=bagatelle.waypoint_frames,
            waypoint_target_keys=bagatelle.waypoint_target_keys,
            waypoint_hand_joints=bagatelle.waypoint_hand_joints,
            config=planner_config,
            boost_height=0.028,
            boost_duration_s=0.12,
            start_after_hold_s=0.005,
        )

    substeps = int(planner_config.interpolation_substeps)
    dense_dt = float(planner_config.control_timestep) / float(substeps)
    target_keys_dense = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    trajectory_path = run_dir / "trajectory.npz"
    atomic_save_npz(
        trajectory_path,
        target_keys=target_keys,
        target_keys_dense=target_keys_dense,
        waypoint_frames=bagatelle.waypoint_frames,
        waypoint_target_keys=bagatelle.waypoint_target_keys,
        waypoint_hand_joints=sanitized,
        bagatelle_width2_waypoint_hand_joints=bagatelle.waypoint_hand_joints,
        planned_hand_joints_dense_pre_magnet=dense,
        planned_hand_joints_dense=dense_aniso,
        segment_ids_dense=np.repeat(np.arange(target_keys.shape[0], dtype=np.int32), substeps)[: dense_aniso.shape[0]],
    )

    played, runtime = rollout_dense(dense_aniso, target_keys_dense, run_dir, f"maestro_liszt_preludio_{condition_label}_200hz", dense_dt, render=True)
    steps = min(int(played.shape[0]), int(target_keys_dense.shape[0]))
    score = score_rollout(
        target_keys=target_keys_dense[:steps],
        played_keys=played[:steps],
        dt=dense_dt,
        threshold=0.5,
        timing_tolerance_s=0.15,
    )
    rollout_path = run_dir / f"maestro_liszt_preludio_{condition_label}_200hz_online_rollout.npz"
    atomic_save_npz(rollout_path, target_keys=target_keys_dense[:steps], played_keys=played[:steps])
    write_key_events_csv(run_dir / "robot_key_events.csv", played[:steps], dt=dense_dt, threshold=0.5)
    summary = {
        "run_dir": str(run_dir),
        "condition_label": condition_label,
        "midi_path": str(MAESTRO_MIDI),
        "midi": {
            "canonical_composer": "Franz Liszt",
            "canonical_title": "Transcendental Etude No. 1, Preludio",
            "split": "validation",
            "duration_s": 45.1552083333,
            "source": "MAESTRO v3.0.0",
        },
        "midi_quantization": quant_meta,
        "max_duration_s": None,
        "bagatelle_npz": str(bagatelle_npz),
        "trajectory_npz": str(trajectory_path),
        "rollout_npz": str(rollout_path),
        "bagatelle_config": bagatelle_config.to_dict(),
        "bagatelle_metadata": bagatelle.metadata,
        "planner_config": asdict(planner_config),
        "anisotropic_magnet": {
            "radii_xyz_m": ANISOTROPIC_RADII_XYZ.astype(float).tolist(),
            "target": "Bagatelle key_press_targets",
            "coordinate_frame": "RoboPianist world xyz",
            **magnet_stats,
            "segment_endpoint_z": z_segment_stats,
            "depress_release_lift": release_lift_stats,
        },
        "dense_control_timestep": dense_dt,
        "fps": 200,
        "score": score,
        **runtime,
    }
    atomic_save_json(run_dir / "render_summary.json", summary)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "video_path": runtime.get("video_path"),
                "f1": score["frame_f1"],
                "precision": score["frame_precision"],
                "recall": score["frame_recall"],
                "matched": f"{score['matched_press_events']}/{score['target_press_events']}",
                "missed": score["missed_key_presses"],
                "mispresses": score["mispresses"],
                "magnet_stats": magnet_stats,
            },
            indent=2,
        ),
        flush=True,
    )


def apply_anisotropic_magnet(
    dense: np.ndarray,
    *,
    waypoint_frames: np.ndarray,
    waypoint_assignments: np.ndarray,
    key_press_targets: np.ndarray,
    kinematics: BagatelleKinematics,
    config: PlannerConfig,
    radii_xyz: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    corrected = np.asarray(dense, dtype=np.float32).copy()
    substeps = int(config.interpolation_substeps)
    approach = int(round(float(config.press_approach_s) / (float(config.control_timestep) / substeps)))
    hold = int(round(float(config.press_hold_s) / (float(config.control_timestep) / substeps)))
    applied_frames = 0
    active_pairs = 0
    max_delta = 0.0
    radii = np.asarray(radii_xyz, dtype=np.float32)
    for waypoint_index, frame in enumerate(np.asarray(waypoint_frames, dtype=np.int64)):
        center = int(frame) * substeps
        start = max(0, center - approach)
        end = min(corrected.shape[0] - 1, center + hold)
        assignments = np.asarray(waypoint_assignments[waypoint_index], dtype=np.int32)
        fingers = np.flatnonzero(assignments >= 0).astype(np.int64)
        if fingers.size == 0:
            continue
        keys = assignments[fingers].astype(np.int64)
        targets = np.asarray(key_press_targets[keys], dtype=np.float32)
        for step in range(start, end + 1):
            if step == center and bool(config.preserve_waypoint_endpoints):
                continue
            q0 = corrected[step].copy()
            q1, pairs = solve_anisotropic_3d_correction(
                kinematics,
                q0,
                fingers=fingers,
                targets=targets,
                radii_xyz=radii,
                frame_offset=float(step - center),
                approach_frames=approach,
                hold_frames=hold,
                config=config,
            )
            corrected[step] = q1
            if pairs:
                applied_frames += 1
                active_pairs += pairs
                max_delta = max(max_delta, float(np.linalg.norm(q1 - q0)))
    return corrected.astype(np.float32), {
        "applied_frames": int(applied_frames),
        "active_finger_key_pairs": int(active_pairs),
        "max_joint_delta_norm": float(max_delta),
        "approach_frames": int(approach),
        "hold_frames": int(hold),
    }


def apply_segment_endpoint_z_magnet(
    dense: np.ndarray,
    *,
    waypoint_frames: np.ndarray,
    waypoint_assignments: np.ndarray,
    key_press_targets: np.ndarray,
    kinematics: BagatelleKinematics,
    config: PlannerConfig,
    z_radius: float,
    endpoint_gain: float,
    mid_gain: float,
    endpoint_width_fraction: float,
    z_step_limit: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    corrected = np.asarray(dense, dtype=np.float32).copy()
    substeps = int(config.interpolation_substeps)
    frames = np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)
    applied_frames = 0
    active_pairs = 0
    max_delta = 0.0
    for segment_index in range(max(int(frames.size) - 1, 0)):
        start = int(frames[segment_index]) * substeps
        end = int(frames[segment_index + 1]) * substeps
        if end <= start:
            continue
        endpoints = (
            (segment_index, "start"),
            (segment_index + 1, "end"),
        )
        for step in range(start, end + 1):
            u = float(step - start) / float(end - start)
            q0 = corrected[step].copy()
            for waypoint_index, side in endpoints:
                assignments = np.asarray(waypoint_assignments[waypoint_index], dtype=np.int32)
                fingers = np.flatnonzero(assignments >= 0).astype(np.int64)
                if fingers.size == 0:
                    continue
                keys = assignments[fingers].astype(np.int64)
                targets_z = np.asarray(key_press_targets[keys], dtype=np.float32)[:, 2]
                endpoint_u = u if side == "start" else 1.0 - u
                time_gain = z_endpoint_temporal_gain(
                    endpoint_u,
                    endpoint_gain=endpoint_gain,
                    mid_gain=mid_gain,
                    endpoint_width_fraction=endpoint_width_fraction,
                )
                q1, pair_count = solve_z_endpoint_delta(
                    kinematics,
                    corrected[step],
                    fingers,
                    targets_z,
                    z_radius=float(z_radius),
                    time_gain=float(time_gain),
                    damping=float(config.ik_damping),
                    max_delta_q=float(config.ik_max_delta_q),
                    z_step_limit=float(z_step_limit),
                )
                corrected[step] = q1
                active_pairs += int(pair_count)
            if not np.allclose(q0, corrected[step]):
                applied_frames += 1
                max_delta = max(max_delta, float(np.linalg.norm(corrected[step] - q0)))
    return corrected.astype(np.float32), {
        "mode": "selected_fingertip_joint_z_only",
        "applied_frames": int(applied_frames),
        "active_finger_key_pairs": int(active_pairs),
        "max_joint_delta_norm": float(max_delta),
        "endpoint_gain": float(endpoint_gain),
        "mid_gain": float(mid_gain),
        "endpoint_width_fraction": float(endpoint_width_fraction),
        "z_step_limit": float(z_step_limit),
    }


def apply_depress_release_lift_boost(
    dense: np.ndarray,
    *,
    waypoint_frames: np.ndarray,
    waypoint_target_keys: np.ndarray,
    waypoint_hand_joints: np.ndarray,
    config: PlannerConfig,
    boost_height: float,
    boost_duration_s: float,
    start_after_hold_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    corrected = np.asarray(dense, dtype=np.float32).copy()
    substeps = int(config.interpolation_substeps)
    dense_dt = float(config.control_timestep) / float(substeps)
    hold_frames = int(round(float(config.press_hold_s) / dense_dt))
    delay_frames = int(round(float(start_after_hold_s) / dense_dt))
    boost_frames = max(int(round(float(boost_duration_s) / dense_dt)), 1)
    applied_frames = 0
    max_delta = 0.0
    for waypoint_index, frame in enumerate(np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)):
        center = int(frame) * substeps
        hands = keyset_hand_sides(waypoint_target_keys[waypoint_index], threshold=float(config.threshold))
        for _side, vertical_index in (("right", RIGHT_FOREARM_TY_INDEX), ("left", LEFT_FOREARM_TY_INDEX)):
            if not hands.get(_side, False):
                continue
            pressed = float(
                np.clip(
                    float(waypoint_hand_joints[waypoint_index, vertical_index]) - float(config.press_depth),
                    float(config.vertical_min),
                    float(config.vertical_max),
                )
            )
            target_clearance = float(np.clip(pressed + float(boost_height), float(config.vertical_min), float(config.vertical_max)))
            start = max(0, center + hold_frames + delay_frames)
            end = min(corrected.shape[0] - 1, start + boost_frames)
            for step in range(start, end + 1):
                u = float(step - start) / float(max(end - start, 1))
                # Fast early lift, then taper so we do not create a discontinuity at the end of the boost.
                weight = 1.0 - float(np.exp(-6.0 * u))
                before = float(corrected[step, vertical_index])
                corrected[step, vertical_index] = max(before, before * (1.0 - weight) + target_clearance * weight)
                corrected[step, vertical_index] = float(
                    np.clip(corrected[step, vertical_index], float(config.vertical_min), float(config.vertical_max))
                )
                delta = abs(float(corrected[step, vertical_index]) - before)
                if delta > 1e-8:
                    applied_frames += 1
                    max_delta = max(max_delta, delta)
    return corrected.astype(np.float32), {
        "mode": "forearm_ty_upward_release_boost",
        "boost_height": float(boost_height),
        "boost_duration_s": float(boost_duration_s),
        "start_after_hold_s": float(start_after_hold_s),
        "applied_vertical_frames": int(applied_frames),
        "max_vertical_delta": float(max_delta),
    }


def z_endpoint_temporal_gain(
    distance_from_endpoint_u: float,
    *,
    endpoint_gain: float,
    mid_gain: float,
    endpoint_width_fraction: float,
) -> float:
    width = max(float(endpoint_width_fraction), 1e-6)
    endpoint = np.exp(-float(distance_from_endpoint_u) / width)
    return float(mid_gain) + float(endpoint_gain) * float(endpoint)


def solve_z_endpoint_delta(
    kinematics: BagatelleKinematics,
    q: np.ndarray,
    fingers: np.ndarray,
    target_z: np.ndarray,
    *,
    z_radius: float,
    time_gain: float,
    damping: float,
    max_delta_q: float,
    z_step_limit: float,
) -> tuple[np.ndarray, int]:
    current = kinematics.fingertip_positions_for_qpos(q)[fingers, 2]
    z_error = np.asarray(target_z, dtype=np.float32).reshape(-1) - current
    z_distance = np.abs(z_error)
    weights = magnet_decay_weight(z_distance, radius=float(z_radius)) * float(time_gain)
    if not np.any(weights > 0.0):
        return np.asarray(q, dtype=np.float32).copy(), 0
    desired_z = current.copy()
    active = 0
    for index, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        step_z = float(z_error[index]) * float(weight)
        step_z = float(np.clip(step_z, -float(z_step_limit), float(z_step_limit)))
        desired_z[index] = current[index] + step_z
        active += 1
    out = solve_z_delta_selected_finger_joints(
        kinematics,
        q,
        fingers,
        desired_z,
        damping=damping,
        max_delta_q=max_delta_q,
    )
    return out.astype(np.float32), active


def solve_anisotropic_3d_correction(
    kinematics: BagatelleKinematics,
    q: np.ndarray,
    *,
    fingers: np.ndarray,
    targets: np.ndarray,
    radii_xyz: np.ndarray,
    frame_offset: float,
    approach_frames: int,
    hold_frames: int,
    config: PlannerConfig,
) -> tuple[np.ndarray, int]:
    out = np.asarray(q, dtype=np.float32).reshape(-1).copy()
    active_count = 0
    for _ in range(max(int(config.ik_iterations_per_frame), 1)):
        current = kinematics.fingertip_positions_for_qpos(out)[fingers]
        vectors = np.asarray(targets, dtype=np.float32) - current
        time_weight = float(
            frame_window_envelope(
                frame_offset,
                approach=int(approach_frames),
                hold=int(hold_frames),
                release=0,
                power=float(config.magnet_power),
            )
        )

        xy_vectors = vectors[:, :2]
        xy_distance = np.linalg.norm(xy_vectors, axis=1)
        xy_weights = magnet_decay_weight(xy_distance, radius=float(np.max(radii_xyz[:2]))) * time_weight * float(config.magnet_gain)

        z_radius = float(radii_xyz[2])
        z_vectors = vectors[:, 2]
        z_distance = np.abs(z_vectors)
        z_weights = magnet_decay_weight(z_distance, radius=z_radius) * time_weight * float(config.magnet_gain)

        if not np.any(xy_weights > 0.0) and not np.any(z_weights > 0.0):
            return out.astype(np.float32), active_count

        if np.any(xy_weights > 0.0):
            desired_xy = current[:, :2].copy()
            for i, weight in enumerate(xy_weights):
                if weight <= 0.0:
                    continue
                step_vec = xy_vectors[i] * float(weight)
                norm = float(np.linalg.norm(step_vec))
                limit = max(float(config.magnet_max_xy_step), 0.0)
                if norm > limit > 0.0:
                    step_vec *= limit / norm
                desired_xy[i] = current[i, :2] + step_vec
                active_count += 1
            out = solve_xy_delta(kinematics, out, fingers, desired_xy, damping=float(config.ik_damping), max_delta_q=float(config.ik_max_delta_q))

        if np.any(z_weights > 0.0):
            current_after_xy = kinematics.fingertip_positions_for_qpos(out)[fingers]
            desired_z = current_after_xy[:, 2].copy()
            for i, weight in enumerate(z_weights):
                if weight <= 0.0:
                    continue
                step_z = float(targets[i, 2] - current_after_xy[i, 2]) * float(weight)
                limit = max(float(config.magnet_max_xy_step), 0.0)
                step_z = float(np.clip(step_z, -limit, limit))
                desired_z[i] = current_after_xy[i, 2] + step_z
                active_count += 1
            out = solve_z_delta_selected_finger_joints(
                kinematics,
                out,
                fingers,
                desired_z,
                damping=float(config.ik_damping),
                max_delta_q=float(config.ik_max_delta_q),
            )
    return out.astype(np.float32), active_count


def solve_xy_delta(
    kinematics: BagatelleKinematics,
    q: np.ndarray,
    fingers: np.ndarray,
    target_xy: np.ndarray,
    *,
    damping: float,
    max_delta_q: float,
) -> np.ndarray:
    base = np.asarray(q, dtype=np.float32).reshape(-1).copy()
    current = kinematics.fingertip_positions_for_qpos(base)[fingers, :2].reshape(-1)
    error = (np.asarray(target_xy, dtype=np.float32).reshape(-1) - current).astype(np.float32)
    if float(np.linalg.norm(error)) <= 1e-8:
        return base
    eps = 1e-4
    jac = np.zeros((error.size, base.size), dtype=np.float32)
    for joint_index in range(base.size):
        perturbed = base.copy()
        perturbed[joint_index] += eps
        xy = kinematics.fingertip_positions_for_qpos(perturbed)[fingers, :2].reshape(-1)
        jac[:, joint_index] = (xy - current) / eps
    return solve_damped_delta(kinematics, base, jac, error, damping=damping, max_delta_q=max_delta_q)


def solve_z_delta_selected_finger_joints(
    kinematics: BagatelleKinematics,
    q: np.ndarray,
    fingers: np.ndarray,
    target_z: np.ndarray,
    *,
    damping: float,
    max_delta_q: float,
) -> np.ndarray:
    base = np.asarray(q, dtype=np.float32).reshape(-1).copy()
    current = kinematics.fingertip_positions_for_qpos(base)[fingers, 2].reshape(-1)
    error = (np.asarray(target_z, dtype=np.float32).reshape(-1) - current).astype(np.float32)
    if float(np.linalg.norm(error)) <= 1e-8:
        return base

    allowed = sorted({joint for finger in fingers.astype(int) for joint in finger_joint_indices(int(finger))})
    if not allowed:
        return base
    eps = 1e-4
    jac_small = np.zeros((error.size, len(allowed)), dtype=np.float32)
    for col, joint_index in enumerate(allowed):
        perturbed = base.copy()
        perturbed[joint_index] += eps
        z = kinematics.fingertip_positions_for_qpos(perturbed)[fingers, 2].reshape(-1)
        jac_small[:, col] = (z - current) / eps

    lhs = jac_small @ jac_small.T + (float(damping) ** 2) * np.eye(jac_small.shape[0], dtype=np.float32)
    try:
        delta_small = jac_small.T @ np.linalg.solve(lhs, error)
    except np.linalg.LinAlgError:
        delta_small = np.zeros((len(allowed),), dtype=np.float32)
    delta = np.zeros_like(base)
    delta[np.asarray(allowed, dtype=np.int64)] = delta_small.astype(np.float32)
    norm = float(np.linalg.norm(delta))
    limit = max(float(max_delta_q), 0.0)
    if norm > limit > 0.0:
        delta *= limit / norm
    return kinematics.clip_qpos(base + delta.astype(np.float32))


def magnet_decay_weight(
    distance_m: np.ndarray,
    *,
    radius: float,
    soft_start: float = 0.06,
    sigma: float = 0.045,
) -> np.ndarray:
    distance = np.asarray(distance_m, dtype=np.float32)
    radius_value = max(float(radius), 1e-8)
    core = np.exp(-np.square(distance / max(float(sigma), 1e-8))).astype(np.float32)
    taper = np.ones_like(core, dtype=np.float32)
    far = distance > float(soft_start)
    taper[far] = np.square(
        np.clip((radius_value - distance[far]) / max(radius_value - float(soft_start), 1e-8), 0.0, 1.0)
    ).astype(np.float32)
    return np.where((distance >= 0.0) & (distance <= radius_value), core * taper, 0.0).astype(np.float32)


def finger_joint_indices(finger: int) -> tuple[int, ...]:
    # Bagatelle fingertip order: left thumb/index/middle/ring/little, then right thumb/index/middle/ring/little.
    # Reduced joint order: right hand 0..22, left hand 23..45.
    mapping = {
        0: (41, 42, 43),          # left thumb
        1: (25, 26, 27, 28),      # left index
        2: (29, 30, 31, 32),      # left middle
        3: (33, 34, 35, 36),      # left ring
        4: (37, 38, 39, 40),      # left little
        5: (18, 19, 20),          # right thumb
        6: (2, 3, 4, 5),          # right index
        7: (6, 7, 8, 9),          # right middle
        8: (10, 11, 12, 13),      # right ring
        9: (14, 15, 16, 17),      # right little
    }
    return mapping.get(int(finger), ())


def solve_damped_delta(
    kinematics: BagatelleKinematics,
    base: np.ndarray,
    jac: np.ndarray,
    error: np.ndarray,
    *,
    damping: float,
    max_delta_q: float,
) -> np.ndarray:
    lhs = jac @ jac.T + (float(damping) ** 2) * np.eye(jac.shape[0], dtype=np.float32)
    try:
        delta = jac.T @ np.linalg.solve(lhs, error)
    except np.linalg.LinAlgError:
        delta = np.zeros_like(base)
    norm = float(np.linalg.norm(delta))
    limit = max(float(max_delta_q), 0.0)
    if norm > limit > 0.0:
        delta *= limit / norm
    return kinematics.clip_qpos(base + delta.astype(np.float32))


def rollout_dense(hand_states: np.ndarray, target_keys_dense: np.ndarray, output_dir: Path, label: str, dt: float, *, render: bool) -> tuple[np.ndarray, dict[str, Any]]:
    midi_proto = write_goals_proto(target_keys_dense[:, :88], output_dir / f"{label}_target_goals.proto", dt=dt, title=label)
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names("RoboPianist-debug-TwinkleTwinkleLittleStar-v0"),
        midi_proto_path=midi_proto,
        control_timestep=dt,
        seed=0,
        reduced_action_space=True,
        extra_task_kwargs={"disable_forearm_reward": True, "disable_fingering_reward": True, "disable_colorization": True, "disable_hand_collisions": False, "wrong_press_termination": False},
        suite_load_kwargs=None,
        prefer_canonical_midi=False,
    )
    played_roll: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    render_error = None
    terminated = False
    restored_count = 0
    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        for hand_state in hand_states:
            restored_count = _set_reduced_hand_qpos(task, physics, hand_state)
            if hasattr(physics, "forward"):
                physics.forward()
            timestep = env.step(zero_action)
            update_key_state = getattr(piano, "_update_key_state", None)
            if callable(update_key_state):
                update_key_state(physics)
            update_key_color = getattr(piano, "_update_key_color", None)
            if callable(update_key_color):
                update_key_color(physics)
            activation = _capture_piano_activation(env)
            if activation is None:
                activation = np.asarray(getattr(piano, "activation"), dtype=np.float32).reshape(-1)[:88]
            played_roll.append(np.asarray(activation[:88], dtype=np.float32))
            if render and render_error is None:
                try:
                    frames.append(render_frame(env, height=480, width=640))
                except Exception as exc:
                    render_error = str(exc)
            if timestep.last():
                terminated = True
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    played = np.stack(played_roll, axis=0) if played_roll else np.zeros((0, 88), dtype=np.float32)
    runtime: dict[str, Any] = {
        "environment_name": env_name,
        "load_info": load_info,
        "midi_proto_path": str(midi_proto),
        "terminated": bool(terminated),
        "restored_hand_joint_count": int(restored_count),
        "rendered_frames": int(len(frames)),
        "render_error": render_error,
    }
    if render and render_error is None:
        events = piano_roll_to_midi_events(played, dt=dt, threshold=0.5)
        video_path, video_format, video_audio_warning = write_video(frames, output_dir / "rollout_video.mp4", fps=200, audio_events=events)
        runtime.update({"video_path": str(video_path), "video_format": video_format, "video_audio_warning": video_audio_warning})
    return played, runtime


def write_key_events_csv(path: Path, piano_roll: np.ndarray, *, dt: float, threshold: float) -> None:
    active = np.asarray(piano_roll, dtype=np.float32)[:, :88] > float(threshold)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "step", "event", "key_index", "midi_note"])
        writer.writeheader()
        for key in range(active.shape[1]):
            was_active = False
            for step, is_active in enumerate(active[:, key]):
                if bool(is_active) and not was_active:
                    writer.writerow({"time_s": float(step * dt), "step": int(step), "event": "note_on", "key_index": int(key), "midi_note": int(21 + key)})
                elif was_active and not bool(is_active):
                    writer.writerow({"time_s": float(step * dt), "step": int(step), "event": "note_off", "key_index": int(key), "midi_note": int(21 + key)})
                was_active = bool(is_active)
            if was_active:
                writer.writerow({"time_s": float(active.shape[0] * dt), "step": int(active.shape[0]), "event": "note_off", "key_index": int(key), "midi_note": int(21 + key)})


if __name__ == "__main__":
    main()
