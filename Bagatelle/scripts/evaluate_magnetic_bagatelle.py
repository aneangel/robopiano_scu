#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir  # noqa: E402
from intermezzo.online_eval import RolloutConfig, score_rollout  # noqa: E402
from intermezzo.planner import PlannerConfig, compute_hand_velocities, plan_between_waypoints  # noqa: E402
from intermezzo.kinematics import RoboPianistHandKinematics  # noqa: E402
from intermezzo.fingertip_assignment import assign_active_fingertips  # noqa: E402
from partita.evaluation.rollout import (  # noqa: E402
    _capture_piano_activation,
    _load_env,
    _locate_task_physics_piano,
    _set_reduced_hand_qpos,
    candidate_environment_names,
    render_frame,
    write_goals_proto,
    write_video,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Intermezzo magnetic key attraction to a Bagatelle trajectory and evaluate online.")
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", default="bagatelle_magnetic_eval")
    parser.add_argument("--label", default="magnetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--timing-tolerance-s", type=float, default=0.15)
    parser.add_argument("--magnet-radius", type=float, default=0.12)
    parser.add_argument("--magnet-sigma", type=float, default=0.06)
    parser.add_argument("--magnet-gain", type=float, default=1.5)
    parser.add_argument("--magnet-max-xy-step", type=float, default=0.01)
    parser.add_argument("--magnet-start-fraction", type=float, default=0.40)
    parser.add_argument("--magnet-power", type=float, default=1.0)
    parser.add_argument("--ik-damping", type=float, default=1e-3)
    parser.add_argument("--ik-max-delta-q", type=float, default=0.03)
    parser.add_argument("--ik-iterations-per-frame", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--selected-finger-z-attraction", action="store_true")
    parser.add_argument("--use-dense", action="store_true", default=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser


def load_payload(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def magnetic_plan(payload: dict[str, np.ndarray], args: argparse.Namespace, run_dir: Path) -> dict[str, np.ndarray]:
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)
    waypoint_frames = np.asarray(payload["waypoint_frames"], dtype=np.int64)
    waypoint_target_keys = np.asarray(payload["waypoint_target_keys"], dtype=np.float32)
    waypoint_hand_joints = np.asarray(payload["waypoint_hand_joints"], dtype=np.float32)
    bag_cfg = BagatelleConfig(
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
        environment_name=str(args.environment_name),
        seed=int(args.seed),
    )
    with BagatelleKinematics(bag_cfg, target_keys=target_keys, output_dir=run_dir / "magnet_kinematics") as kin:
        key_geometry = kin.key_press_targets(np.arange(88, dtype=np.int32))[:, :2].astype(np.float32)
        rp_kin = RoboPianistHandKinematics(kin.task, kin.physics)
        cfg = PlannerConfig(
            control_timestep=float(args.control_timestep),
            threshold=float(args.threshold),
            clearance_height=float(bag_cfg.clearance_height),
            lift_fraction=float(bag_cfg.lift_fraction),
            descent_fraction=float(bag_cfg.descent_fraction),
            vertical_min=float(bag_cfg.vertical_min),
            vertical_max=float(bag_cfg.vertical_max),
            enable_key_magnetism=True,
            magnet_radius=float(args.magnet_radius),
            magnet_sigma=float(args.magnet_sigma),
            magnet_gain=float(args.magnet_gain),
            magnet_max_xy_step=float(args.magnet_max_xy_step),
            magnet_start_fraction=float(args.magnet_start_fraction),
            magnet_power=float(args.magnet_power),
            ik_damping=float(args.ik_damping),
            ik_max_delta_q=float(args.ik_max_delta_q),
            ik_iterations_per_frame=int(args.ik_iterations_per_frame),
            preserve_waypoint_endpoints=True,
            magnet_only_final_keyset=True,
            selected_finger_z_attraction=bool(args.selected_finger_z_attraction),
        )
        planned, velocities, segment_ids, sanitized, dense, dense_velocities, dense_segment_ids = plan_between_waypoints(
            total_steps=int(target_keys.shape[0]),
            waypoint_frames=waypoint_frames,
            waypoint_target_keys=waypoint_target_keys,
            waypoint_hand_joints=waypoint_hand_joints,
            config=cfg,
            key_geometry=key_geometry,
            kinematics=rp_kin,
            return_dense=True,
        )
    out = dict(payload)
    out["planned_hand_joints"] = planned.astype(np.float32)
    out["planned_hand_velocities"] = velocities.astype(np.float32)
    out["segment_ids"] = segment_ids.astype(np.int32)
    out["waypoint_hand_joints"] = sanitized.astype(np.float32)
    out["planned_hand_joints_dense"] = dense.astype(np.float32)
    out["planned_hand_velocities_dense"] = dense_velocities.astype(np.float32)
    out["segment_ids_dense"] = dense_segment_ids.astype(np.int32)
    out["target_keys_dense"] = np.repeat(target_keys, int(cfg.interpolation_substeps), axis=0).astype(np.float32)[: dense.shape[0]]
    out["dense_control_timestep"] = np.asarray(float(cfg.control_timestep) / float(cfg.interpolation_substeps), dtype=np.float32)
    return out



def _key_xy_from_piano(piano: Any, physics: Any) -> np.ndarray:
    positions: list[np.ndarray] = []
    for key in piano.keys[:88]:
        key_geom = key.geom[0]
        pos = np.asarray(physics.bind(key_geom).xpos, dtype=np.float32).copy()
        size = np.asarray(physics.bind(key_geom).size, dtype=np.float32)
        pos[0] += 0.35 * float(size[0])
        positions.append(pos[:2].astype(np.float32))
    return np.stack(positions, axis=0).astype(np.float32)


def _float_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean_m": None, "median_m": None, "p90_m": None, "p95_m": None, "max_m": None}
    return {
        "count": int(arr.size),
        "mean_m": float(np.mean(arr)),
        "median_m": float(np.median(arr)),
        "p90_m": float(np.percentile(arr, 90)),
        "p95_m": float(np.percentile(arr, 95)),
        "max_m": float(np.max(arr)),
        "success_at_0p005": float(np.mean(arr <= 0.005)),
        "success_at_0p01": float(np.mean(arr <= 0.010)),
        "success_at_0p02": float(np.mean(arr <= 0.020)),
        "success_at_0p05": float(np.mean(arr <= 0.050)),
    }


def summarize_selected_fingertip_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selected_pairs": int(len(rows)),
        "xy_distance": _float_summary([row["xy_distance_m"] for row in rows]),
        "x_distance": _float_summary([row["x_distance_m"] for row in rows]),
        "y_distance": _float_summary([row["y_distance_m"] for row in rows]),
    }


def rollout_and_render(hand_targets: np.ndarray, target_keys: np.ndarray, run_dir: Path, args: argparse.Namespace, *, control_timestep: float | None = None) -> dict[str, Any]:
    output = run_dir / "rollout"
    output.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(hand_targets, dtype=np.float32)
    keys = np.asarray(target_keys, dtype=np.float32)[:, :88]
    steps = min(int(targets.shape[0]), int(keys.shape[0]))
    dt = float(args.control_timestep if control_timestep is None else control_timestep)
    midi_proto = write_goals_proto(keys[:steps], output / f"{args.label}_target_goals.proto", dt=dt, title=f"Bagatelle magnetic {args.label}")
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(str(args.environment_name)),
        midi_proto_path=midi_proto,
        control_timestep=dt,
        seed=int(args.seed),
        reduced_action_space=True,
        extra_task_kwargs={
            "disable_forearm_reward": True,
            "disable_fingering_reward": True,
            "disable_colorization": True,
            "disable_hand_collisions": False,
            "wrong_press_termination": False,
        },
        suite_load_kwargs=None,
        prefer_canonical_midi=False,
    )
    played_rows: list[np.ndarray] = []
    fingertip_distance_rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    render_error = None
    restored_count = 0
    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        metric_kinematics = RoboPianistHandKinematics(task, physics)
        metric_key_xy = _key_xy_from_piano(piano, physics)
        for step in range(steps):
            restored_count = _set_reduced_hand_qpos(task, physics, targets[step])
            if hasattr(physics, "forward"):
                physics.forward()
            for _ in range(max(int(args.settle_steps), 0)):
                restored_count = _set_reduced_hand_qpos(task, physics, targets[step])
                if hasattr(physics, "step"):
                    physics.step()
                elif hasattr(physics, "forward"):
                    physics.forward()
            if hasattr(physics, "forward"):
                physics.forward()
            update_key_state = getattr(piano, "_update_key_state", None)
            if callable(update_key_state):
                update_key_state(physics)
            update_key_color = getattr(piano, "_update_key_color", None)
            if callable(update_key_color):
                update_key_color(physics)
            activation = _capture_piano_activation(env)
            if activation is not None:
                played_rows.append(np.asarray(activation[:88], dtype=np.float32))
            active_row = keys[step]
            if np.any(active_row[:88] > float(args.threshold)):
                assignments = assign_active_fingertips(
                    active_row,
                    endpoint_hand_state=targets[step],
                    key_xy=metric_key_xy,
                    kinematics=metric_kinematics,
                    threshold=float(args.threshold),
                )
                actual_xy = metric_kinematics.fingertip_xy()
                for item in assignments:
                    finger = int(item.fingertip_index)
                    key = int(item.key_index)
                    delta = actual_xy[finger] - metric_key_xy[key]
                    fingertip_distance_rows.append(
                        {
                            "frame": int(step),
                            "time_s": float(step * dt),
                            "finger_index": finger,
                            "key_index": key,
                            "x_distance_m": float(abs(delta[0])),
                            "y_distance_m": float(abs(delta[1])),
                            "xy_distance_m": float(np.linalg.norm(delta)),
                        }
                    )
            if bool(args.render) and step % max(int(args.render_every), 1) == 0 and render_error is None:
                try:
                    frames.append(render_frame(env, height=int(args.height), width=int(args.width)))
                except Exception as exc:
                    render_error = str(exc)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    played = np.stack(played_rows, axis=0) if played_rows else np.zeros((0, 88), dtype=np.float32)
    score = score_rollout(
        target_keys=keys[: played.shape[0]],
        played_keys=played,
        dt=dt,
        threshold=float(args.threshold),
        timing_tolerance_s=float(args.timing_tolerance_s),
    )
    video_path = None
    video_format = None
    audio_warning = None
    if bool(args.render) and render_error is None and frames:
        video_path, video_format, audio_warning = write_video(
            frames,
            output / f"{args.label}_magnetic_rollout.mp4",
            fps=max(int(args.fps / max(int(args.render_every), 1)), 1),
        )
    atomic_save_npz(output / f"{args.label}_online_rollout.npz", target_keys=keys[: played.shape[0]], played_keys=played)
    if fingertip_distance_rows:
        import csv

        with (output / f"{args.label}_selected_fingertip_key_distances.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["frame", "time_s", "finger_index", "key_index", "x_distance_m", "y_distance_m", "xy_distance_m"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fingertip_distance_rows)
    result = {
        "label": str(args.label),
        "environment_name": env_name,
        "load_info": load_info,
        "midi_proto_path": str(midi_proto),
        "hand_targets_shape": list(targets.shape),
        "target_keys_shape": list(keys.shape),
        "played_keys_shape": list(played.shape),
        "pose_frames_applied": int(steps),
        "restored_hand_joint_count": int(restored_count),
        "score": score,
        "selected_fingertip_key_distance": summarize_selected_fingertip_rows(fingertip_distance_rows),
        "settle_steps": int(args.settle_steps),
        "rendered_frames": int(len(frames)),
        "render_every": int(args.render_every),
        "render_error": render_error,
        "video_path": str(video_path) if video_path is not None else None,
        "video_format": video_format,
        "audio_warning": audio_warning,
    }
    atomic_save_json(output / f"{args.label}_online_rollout.json", result)
    return result


def main() -> None:
    args = build_parser().parse_args()
    payload = load_payload(args.trajectory_npz)
    run_dir = create_unique_run_dir(Path(args.output_root).expanduser(), run_name=args.run_name, prefix="bagatelle_magnetic")
    magnetic = magnetic_plan(payload, args, run_dir)
    trajectory_path = run_dir / "trajectory_magnetic.npz"
    atomic_save_npz(trajectory_path, **magnetic)
    if bool(args.use_dense):
        hand_targets = magnetic["planned_hand_joints_dense"]
        target_keys = magnetic["target_keys_dense"]
        rollout_dt = float(np.asarray(magnetic["dense_control_timestep"]))
    else:
        hand_targets = magnetic["planned_hand_joints"]
        target_keys = magnetic["target_keys"]
        rollout_dt = float(args.control_timestep)
    rollout = rollout_and_render(hand_targets, target_keys, run_dir, args, control_timestep=rollout_dt)
    summary = {
        "run_dir": str(run_dir),
        "source_trajectory_npz": str(Path(args.trajectory_npz).expanduser().resolve()),
        "magnetic_trajectory_npz": str(trajectory_path),
        "magnet_config": {
            "magnet_radius": float(args.magnet_radius),
            "magnet_sigma": float(args.magnet_sigma),
            "magnet_gain": float(args.magnet_gain),
            "magnet_max_xy_step": float(args.magnet_max_xy_step),
            "magnet_start_fraction": float(args.magnet_start_fraction),
            "magnet_power": float(args.magnet_power),
            "ik_damping": float(args.ik_damping),
            "ik_max_delta_q": float(args.ik_max_delta_q),
            "ik_iterations_per_frame": int(args.ik_iterations_per_frame),
            "selected_finger_z_attraction": bool(args.selected_finger_z_attraction),
        },
        "rollout": rollout,
    }
    atomic_save_json(run_dir / f"{args.label}_magnetic_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote magnetic Bagatelle eval: {run_dir}")


if __name__ == "__main__":
    main()
