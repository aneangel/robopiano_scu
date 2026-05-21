from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bagatelle.config import BagatelleConfig
from bagatelle.kinematics import HAND_STATE_DIM, BagatelleKinematics


DEFAULT_TRAJECTORY = Path(
    "/WAVE/datasets/ccoelho_lab-jlanders/Bagatelle/sequence_eval/lookahead2_twinkle_full/trajectory.npz"
)


def _array_summary(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in data.items()
    ]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_stats(values: np.ndarray, prefix: str) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_mean_m": None,
            f"{prefix}_median_m": None,
            f"{prefix}_p90_m": None,
            f"{prefix}_p95_m": None,
            f"{prefix}_max_m": None,
        }
    return {
        f"{prefix}_mean_m": float(np.mean(finite)),
        f"{prefix}_median_m": float(np.median(finite)),
        f"{prefix}_p90_m": float(np.percentile(finite, 90)),
        f"{prefix}_p95_m": float(np.percentile(finite, 95)),
        f"{prefix}_max_m": float(np.max(finite)),
    }


def _success_rates(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "success_rate_1cm": None,
            "success_rate_2cm": None,
            "success_rate_3cm": None,
            "success_rate_5cm": None,
            "success_rate_10cm": None,
        }
    return {
        "success_rate_1cm": float(np.mean(finite <= 0.01)),
        "success_rate_2cm": float(np.mean(finite <= 0.02)),
        "success_rate_3cm": float(np.mean(finite <= 0.03)),
        "success_rate_5cm": float(np.mean(finite <= 0.05)),
        "success_rate_10cm": float(np.mean(finite <= 0.10)),
    }


def _element_name(element: Any) -> str:
    for attr in ("full_identifier", "identifier", "name"):
        value = getattr(element, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                pass
        if value:
            return str(value)
    return repr(element)


def _qpos_addresses(physics: Any, joints: list[Any]) -> list[int | None]:
    out: list[int | None] = []
    model = getattr(physics, "model", None)
    if model is None:
        return [None for _ in joints]
    for joint in joints:
        name = _element_name(joint)
        candidates = [name, name.split("/")[-1]]
        addr = None
        for candidate in candidates:
            try:
                joint_id = model.name2id(candidate, "joint")
                addr = int(model.jnt_qposadr[joint_id])
                break
            except Exception:
                continue
        out.append(addr)
    return out


def _active_waypoint_indices(data: dict[str, np.ndarray]) -> np.ndarray:
    if "assignments" in data:
        assignments = np.asarray(data["assignments"])
        if assignments.ndim == 2:
            return np.flatnonzero(np.any(assignments >= 0, axis=1)).astype(np.int32)
    if "waypoint_target_keys" in data:
        keys = np.asarray(data["waypoint_target_keys"])
        if keys.ndim == 2:
            return np.flatnonzero(np.any(keys[:, :88] > 0.5, axis=1)).astype(np.int32)
    return np.zeros((0,), dtype=np.int32)


def _selected_waypoints(active_waypoints: np.ndarray, max_frames: int | None) -> np.ndarray:
    if active_waypoints.size == 0:
        return active_waypoints
    if max_frames is None or max_frames <= 0 or active_waypoints.size <= max_frames:
        return active_waypoints
    sample = np.linspace(0, active_waypoints.size - 1, max_frames).round().astype(np.int32)
    return np.unique(active_waypoints[sample]).astype(np.int32)


def _set_hand_qpos_direct(physics: Any, joints: list[Any], values: np.ndarray) -> int:
    hand_values = np.asarray(values, dtype=np.float32).reshape(-1)
    if hand_values.size < len(joints):
        raise ValueError(f"saved hand qpos has {hand_values.size} values, sim expects {len(joints)}")
    for joint, value in zip(joints, hand_values[: len(joints)]):
        physics.bind(joint).qpos = float(value)
    return len(joints)


def _zero_qvel(physics: Any) -> None:
    qvel = getattr(getattr(physics, "data", None), "qvel", None)
    if qvel is not None:
        qvel[:] = 0.0


def _finger_rows_for_waypoint(
    *,
    data: dict[str, np.ndarray],
    waypoint_index: int,
    sim_fingertips: np.ndarray,
    sim_key_targets: np.ndarray,
    key_target_delta: np.ndarray,
    frame: int,
) -> list[dict[str, Any]]:
    assignments = np.asarray(data["assignments"])[waypoint_index]
    offline_fingertips = np.asarray(data["waypoint_fingertips"], dtype=np.float32)[waypoint_index]
    saved_targets = np.asarray(data["fingertip_targets"], dtype=np.float32)[waypoint_index]
    rows: list[dict[str, Any]] = []
    for finger_index, key_index in enumerate(assignments.astype(int).tolist()):
        if key_index < 0:
            continue
        offline_to_saved = float(np.linalg.norm(offline_fingertips[finger_index] - saved_targets[finger_index]))
        sim_to_key = float(np.linalg.norm(sim_fingertips[finger_index] - sim_key_targets[finger_index]))
        sim_to_saved = float(np.linalg.norm(sim_fingertips[finger_index] - saved_targets[finger_index]))
        offline_to_sim = float(np.linalg.norm(offline_fingertips[finger_index] - sim_fingertips[finger_index]))
        rows.append(
            {
                "frame": int(frame),
                "waypoint_index": int(waypoint_index),
                "finger_index": int(finger_index),
                "key_index": int(key_index),
                "offline_fingertip_to_saved_target_m": offline_to_saved,
                "sim_fingertip_to_sim_key_target_m": sim_to_key,
                "sim_fingertip_to_saved_target_m": sim_to_saved,
                "offline_vs_sim_fingertip_delta_m": offline_to_sim,
                "saved_target_vs_sim_key_target_delta_m": float(key_target_delta[finger_index]),
                "offline_fingertip_x": float(offline_fingertips[finger_index, 0]),
                "offline_fingertip_y": float(offline_fingertips[finger_index, 1]),
                "offline_fingertip_z": float(offline_fingertips[finger_index, 2]),
                "sim_fingertip_x": float(sim_fingertips[finger_index, 0]),
                "sim_fingertip_y": float(sim_fingertips[finger_index, 1]),
                "sim_fingertip_z": float(sim_fingertips[finger_index, 2]),
                "sim_key_target_x": float(sim_key_targets[finger_index, 0]),
                "sim_key_target_y": float(sim_key_targets[finger_index, 1]),
                "sim_key_target_z": float(sim_key_targets[finger_index, 2]),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    trajectory_path = Path(args.trajectory).expanduser()
    data = {name: np.asarray(value) for name, value in np.load(trajectory_path, allow_pickle=True).items()}

    print("NPZ keys:")
    for item in _array_summary(data):
        print(f"  {item['name']}: shape={item['shape']} dtype={item['dtype']}")

    state_name = "planned_hand_joints"
    goal_name = "target_keys"
    fingertip_name = "waypoint_fingertips"
    target_name = "fingertip_targets"
    for required in (state_name, goal_name, "waypoint_frames", "assignments", fingertip_name, target_name):
        if required not in data:
            raise KeyError(f"trajectory is missing required array {required!r}")
    if data[state_name].shape[-1] != HAND_STATE_DIM:
        raise ValueError(f"{state_name} last dimension is {data[state_name].shape[-1]}, expected {HAND_STATE_DIM}")

    output_dir = trajectory_path.parent
    json_path = Path(args.json_out) if args.json_out else output_dir / "ik_teleport_verification.json"
    csv_path = Path(args.csv_out) if args.csv_out else output_dir / "ik_teleport_verification_per_frame.csv"

    cfg = BagatelleConfig()
    if args.environment_name:
        cfg = BagatelleConfig(environment_name=args.environment_name)

    active_waypoints = _active_waypoint_indices(data)
    selected_waypoints = _selected_waypoints(active_waypoints, args.max_waypoints)
    waypoint_frames = np.asarray(data["waypoint_frames"], dtype=np.int64)
    print(f"Selected waypoint indices: {selected_waypoints.tolist()}")
    print(f"Selected frame indices: {waypoint_frames[selected_waypoints].astype(int).tolist()}")

    kin = BagatelleKinematics(cfg, target_keys=np.asarray(data[goal_name], dtype=np.float32), output_dir=output_dir)
    per_finger_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    time_changes: list[float] = []
    assigned_count = 0
    restored_count = 0
    try:
        task = kin.task
        physics = kin.physics
        joints = list(kin.joint_handles)
        joint_names = [_element_name(joint) for joint in joints]
        qpos_addresses = _qpos_addresses(physics, joints)
        print(f"Detected hand state array: {state_name}")
        print(f"Detected goal array: {goal_name}")
        print(f"Detected fingertip arrays: saved={fingertip_name}, target={target_name}")
        print(f"Hand qpos joint count: {len(joints)}")
        print(f"Hand qpos joint names: {joint_names}")
        print(f"Hand qpos addresses: {qpos_addresses}")

        for waypoint_index in selected_waypoints.astype(int).tolist():
            frame = int(waypoint_frames[waypoint_index])
            assignments = np.asarray(data["assignments"])[waypoint_index].astype(int)
            active_fingers = np.flatnonzero(assignments >= 0).astype(np.int32)
            active_keys = assignments[active_fingers].astype(np.int32)
            if active_fingers.size == 0:
                continue

            kin.env.reset()
            task = kin.task
            physics = kin.physics
            joints = list(kin.joint_handles)
            time_before = float(physics.data.time)
            restored_count = _set_hand_qpos_direct(physics, joints, np.asarray(data[state_name][frame], dtype=np.float32))
            _zero_qvel(physics)
            time_after_assign = float(physics.data.time)
            physics.forward()
            time_after_forward = float(physics.data.time)
            time_delta = max(abs(time_after_assign - time_before), abs(time_after_forward - time_before))
            time_changes.append(time_delta)

            sim_fingertips_all = kin.current_fingertips()
            sim_fingertips = sim_fingertips_all[active_fingers]
            saved_targets_all = np.asarray(data[target_name], dtype=np.float32)[waypoint_index]
            saved_targets = saved_targets_all[active_fingers]
            sim_key_targets = kin.key_press_targets(active_keys)
            key_target_delta = np.linalg.norm(sim_key_targets - saved_targets, axis=1)

            dense_sim_key_targets = np.full((10, 3), np.nan, dtype=np.float32)
            dense_sim_key_targets[active_fingers] = sim_key_targets
            dense_key_delta = np.full((10,), np.nan, dtype=np.float32)
            dense_key_delta[active_fingers] = key_target_delta
            rows = _finger_rows_for_waypoint(
                data=data,
                waypoint_index=waypoint_index,
                sim_fingertips=sim_fingertips_all,
                sim_key_targets=dense_sim_key_targets,
                key_target_delta=dense_key_delta,
                frame=frame,
            )
            per_finger_rows.extend(rows)
            assigned_count += len(rows)
            frame_rows.append(
                {
                    "frame": frame,
                    "waypoint_index": int(waypoint_index),
                    "active_fingers": active_fingers.astype(int).tolist(),
                    "active_keys": active_keys.astype(int).tolist(),
                    "time_before_forward": time_before,
                    "time_after_forward": time_after_forward,
                    "max_abs_time_change": float(time_delta),
                    "fingertip_positions_after_direct_load": sim_fingertips.astype(float).tolist(),
                    "target_key_positions": sim_key_targets.astype(float).tolist(),
                    "per_fingertip_to_target_distances_m": [row["sim_fingertip_to_sim_key_target_m"] for row in rows],
                }
            )

    finally:
        kin.close()

    sim_dist = np.asarray([row["sim_fingertip_to_sim_key_target_m"] for row in per_finger_rows], dtype=np.float64)
    offline_dist = np.asarray([row["offline_fingertip_to_saved_target_m"] for row in per_finger_rows], dtype=np.float64)
    offline_sim_delta = np.asarray([row["offline_vs_sim_fingertip_delta_m"] for row in per_finger_rows], dtype=np.float64)
    target_delta = np.asarray([row["saved_target_vs_sim_key_target_delta_m"] for row in per_finger_rows], dtype=np.float64)
    max_time_change = float(np.max(time_changes)) if time_changes else 0.0

    report: dict[str, Any] = {
        "trajectory_path": str(trajectory_path),
        "npz_arrays": _array_summary(data),
        "num_frames_tested": int(len(frame_rows)),
        "num_assigned_fingertips_tested": int(assigned_count),
        "selected_waypoint_indices": selected_waypoints.astype(int).tolist(),
        "selected_frame_indices": waypoint_frames[selected_waypoints].astype(int).tolist(),
        "state_array_used": state_name,
        "goal_array_used": goal_name,
        "fingertip_array_used": fingertip_name,
        "fingertip_target_array_used": target_name,
        "sim_qpos_mapping": {
            "method": "physics.bind(task right_hand.joints then left_hand.joints).qpos",
            "joint_names": joint_names,
            "qpos_addresses": qpos_addresses,
        },
        "qpos_directly_assigned_successfully": bool(restored_count == HAND_STATE_DIM and assigned_count > 0),
        "sim_time_before_forward_first_frame": frame_rows[0]["time_before_forward"] if frame_rows else None,
        "sim_time_after_forward_first_frame": frame_rows[0]["time_after_forward"] if frame_rows else None,
        "max_absolute_time_change": max_time_change,
        "sim_time_changed": bool(max_time_change != 0.0),
        "per_frame": frame_rows[: min(len(frame_rows), int(args.max_frames_in_json))],
    }
    report.update(_finite_stats(sim_dist, "fingertip_to_key"))
    report.update(_success_rates(sim_dist))
    report.update(_finite_stats(offline_dist, "offline_fingertip_to_key"))
    report.update(_finite_stats(offline_sim_delta, "offline_vs_sim_fingertip_delta"))
    report.update(_finite_stats(target_delta, "saved_target_vs_sim_key_target_delta"))
    report.update(
        {
            "mean_fingertip_to_key_m": report["fingertip_to_key_mean_m"],
            "median_fingertip_to_key_m": report["fingertip_to_key_median_m"],
            "p90_fingertip_to_key_m": report["fingertip_to_key_p90_m"],
            "p95_fingertip_to_key_m": report["fingertip_to_key_p95_m"],
            "max_fingertip_to_key_m": report["fingertip_to_key_max_m"],
        }
    )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = list(per_finger_rows[0].keys()) if per_finger_rows else [
        "frame",
        "waypoint_index",
        "finger_index",
        "key_index",
        "offline_fingertip_to_saved_target_m",
        "sim_fingertip_to_sim_key_target_m",
        "sim_fingertip_to_saved_target_m",
        "offline_vs_sim_fingertip_delta_m",
        "saved_target_vs_sim_key_target_delta_m",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_finger_rows)

    print(f"qpos directly assigned successfully: {report['qpos_directly_assigned_successfully']}")
    print(f"sim time changed: {report['sim_time_changed']} max_abs_time_change={max_time_change}")
    print(
        "sim fingertip-to-key distances: "
        f"mean={report['fingertip_to_key_mean_m']} "
        f"median={report['fingertip_to_key_median_m']} "
        f"p95={report['fingertip_to_key_p95_m']} "
        f"max={report['fingertip_to_key_max_m']}"
    )
    print(
        "offline fingertip-to-key distances: "
        f"mean={report['offline_fingertip_to_key_mean_m']} "
        f"median={report['offline_fingertip_to_key_median_m']} "
        f"p95={report['offline_fingertip_to_key_p95_m']} "
        f"max={report['offline_fingertip_to_key_max_m']}"
    )
    print(
        "offline-vs-sim fingertip deltas: "
        f"mean={report['offline_vs_sim_fingertip_delta_mean_m']} "
        f"p95={report['offline_vs_sim_fingertip_delta_p95_m']}"
    )
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV: {csv_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Bagatelle saved IK states by direct MuJoCo pose loading.")
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--environment-name", default=None)
    parser.add_argument("--max-waypoints", type=int, default=0, help="0 means evaluate every active waypoint.")
    parser.add_argument("--max-frames-in-json", type=int, default=25)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
