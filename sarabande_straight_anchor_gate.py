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
    REPO_ROOT,
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402
from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.joint_space_trajectory import build_joint_space_straightened_trajectory  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def waypoint_release_frames(target_keys: np.ndarray, waypoint_frames: np.ndarray, *, threshold: float) -> np.ndarray:
    active = np.asarray(target_keys, dtype=np.float32)[:, :88] > float(threshold)
    releases: list[int] = []
    for raw_frame in np.asarray(waypoint_frames, dtype=np.int64).reshape(-1):
        frame = int(np.clip(int(raw_frame), 0, max(active.shape[0] - 1, 0)))
        mask = active[frame]
        end = frame
        while end + 1 < active.shape[0] and np.array_equal(active[end + 1], mask):
            end += 1
        releases.append(end)
    return np.asarray(releases, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sarabande: rebuild Etude/Impromptu contact poses with straight-finger dense anchors."
    )
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--approach-s", type=float, default=0.055)
    parser.add_argument("--release-s", type=float, default=0.025)
    parser.add_argument("--release-fraction", type=float, default=0.20)
    parser.add_argument("--approach-fraction", type=float, default=0.40)
    parser.add_argument("--straight-value", type=float, default=0.0)
    parser.add_argument("--lift-straight-anchors", action="store_true")
    parser.add_argument("--straight-lift-height", type=float, default=0.015)
    parser.add_argument("--straighten-all-fingers", action="store_true")
    parser.add_argument("--no-preserve-sustained-fingers", action="store_true")
    parser.add_argument("--no-straighten-idle-waypoint-fingers", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}

    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control_qpos = np.asarray(payload["planned_hand_joints"], dtype=np.float32)
    dense_base = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32)
    waypoint_frames = np.asarray(payload["waypoint_frames"], dtype=np.int64).reshape(-1)
    assignments = np.asarray(payload["assignments"], dtype=np.int32)
    if waypoint_frames.size == 0:
        raise ValueError("source trajectory has no waypoint_frames")
    if assignments.shape[0] != waypoint_frames.size:
        raise ValueError(f"assignments {assignments.shape} do not align with waypoint_frames {waypoint_frames.shape}")
    substeps = max(int(dense_base.shape[0] // max(control_qpos.shape[0], 1)), 1)
    waypoint_qpos = control_qpos[np.clip(waypoint_frames, 0, max(control_qpos.shape[0] - 1, 0))].astype(np.float32)

    bag_cfg = BagatelleConfig(
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        seed=int(args.seed),
        control_timestep=float(args.control_timestep),
    )
    imp_cfg = ImpromptuConfig(
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
        environment_name=str(args.environment_name),
        interpolation_substeps=int(substeps),
        approach_s=float(args.approach_s),
        release_s=float(args.release_s),
        joint_space_release_fraction=float(args.release_fraction),
        joint_space_approach_fraction=float(args.approach_fraction),
        joint_space_straight_value=float(args.straight_value),
        joint_space_lift_straight_anchors=bool(args.lift_straight_anchors),
        joint_space_straight_lift_height=float(args.straight_lift_height),
        joint_space_straighten_all_fingers=bool(args.straighten_all_fingers),
        joint_space_preserve_sustained_fingers=not bool(args.no_preserve_sustained_fingers),
        joint_space_straighten_idle_fingers_at_waypoints=not bool(args.no_straighten_idle_waypoint_fingers),
    )
    releases = waypoint_release_frames(target_keys, waypoint_frames, threshold=float(args.threshold))
    with BagatelleKinematics(config=bag_cfg, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        neutral = np.asarray(kin.neutral_qpos, dtype=np.float32)
        rebuilt = build_joint_space_straightened_trajectory(
            total_steps=int(target_keys.shape[0]),
            waypoint_frames=waypoint_frames,
            waypoint_release_frames=releases,
            waypoint_qpos=waypoint_qpos,
            assignments=assignments,
            neutral_qpos=neutral,
            config=imp_cfg,
            kinematics=kin,
        )

    dense = np.asarray(rebuilt.dense_qpos, dtype=np.float32)
    planned = dense[::substeps][: target_keys.shape[0]].astype(np.float32)
    dense_dt = float(args.control_timestep) / float(substeps)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)

    payload["waypoint_hand_joints"] = np.asarray(rebuilt.waypoint_qpos, dtype=np.float32)
    payload["planned_hand_joints"] = planned
    payload["planned_hand_velocities"] = compute_hand_velocities(planned, control_timestep=float(args.control_timestep))
    payload["planned_hand_joints_dense"] = dense
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["segment_ids_dense"] = np.asarray(rebuilt.segment_ids, dtype=np.int32)
    atomic_save_npz(out / "trajectory.npz", **payload)

    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(args.environment_name),
        demo_id=0,
        actions=np.zeros((dense.shape[0], 39), dtype=np.float32),
        goals=dense_goals,
        hand_joints=dense,
        environment_name=str(args.environment_name),
    )
    sim_summary = simulate_rp1m_rollout(
        traj,
        RolloutConfig(
            mode="hand_state",
            dataset_timestep=float(dense_dt),
            simulation_timestep=float(dense_dt),
            hand_anchor_y_offset=None,
            hand_state_action_source="zero",
            restore_initial_hand=True,
            set_hand_qvel=False,
            threshold=float(args.threshold),
            render_mp4=False,
            render_audio=False,
        ),
        out / "rp1m_sim",
    )
    with np.load(sim_summary["rollout_npz"], allow_pickle=False) as rollout:
        played = np.asarray(rollout["source_played_piano"], dtype=np.float32)
        goals = np.asarray(rollout["goals"], dtype=np.float32)
    score = score_rollout(
        target_keys=goals,
        played_keys=played,
        dt=float(dense_dt),
        threshold=float(args.threshold),
        timing_tolerance_s=0.15,
    )
    result = {
        "source_npz": str(source),
        "output_dir": str(out),
        "substeps": int(substeps),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "joint_space_metadata": rebuilt.metadata,
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
