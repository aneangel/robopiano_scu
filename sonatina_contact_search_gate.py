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
from bagatelle.kinematics import BagatelleKinematics, HAND_STATE_DIM  # noqa: E402
from impromptu.joint_space_trajectory import BAGATELLE_FINGER_JOINT_INDEX_ROWS  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def contact_counts(target: np.ndarray, activation: np.ndarray, threshold: float) -> tuple[float, int, int, int]:
    goal = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    played = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(goal, played).sum())
    fp = int(np.logical_and(~goal, played).sum())
    fn = int(np.logical_and(goal, ~played).sum())
    denom = 2 * tp + fp + fn
    return (float(2 * tp / denom) if denom else 1.0, tp, fp, fn)


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def interp_rows(anchor_x: np.ndarray, anchor_y: np.ndarray, out_x: np.ndarray) -> np.ndarray:
    anchor_x = np.asarray(anchor_x, dtype=np.float64).reshape(-1)
    anchor_y = np.asarray(anchor_y, dtype=np.float32)
    out_x = np.asarray(out_x, dtype=np.float64).reshape(-1)
    out = np.empty((out_x.size, anchor_y.shape[1]), dtype=np.float32)
    for col in range(anchor_y.shape[1]):
        out[:, col] = np.interp(out_x, anchor_x, anchor_y[:, col]).astype(np.float32)
    return out


def active_joint_indices(assignments: np.ndarray) -> np.ndarray:
    fingers = np.flatnonzero(np.asarray(assignments, dtype=np.int32).reshape(-1) >= 0)
    rows: list[int] = []
    for finger in fingers:
        rows.extend(BAGATELLE_FINGER_JOINT_INDEX_ROWS[int(finger)])
    if not rows:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.asarray(rows, dtype=np.int64)).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Black-box static contact search around Impromptu waypoint qpos.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--samples-per-waypoint", type=int, default=48)
    parser.add_argument("--sigma", type=float, default=0.045)
    parser.add_argument("--min-static-improvement", type=float, default=0.05)
    parser.add_argument("--max-fp-increase", type=int, default=0)
    parser.add_argument("--seed", type=int, default=524)
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    waypoint_frames = np.asarray(payload["waypoint_frames"], dtype=np.int64)
    waypoint_qpos = np.asarray(payload["waypoint_hand_joints"], dtype=np.float32).copy()
    assignments = np.asarray(payload["assignments"], dtype=np.int32)
    control_steps = int(np.asarray(payload["planned_hand_joints"]).shape[0])
    substeps = max(int(np.asarray(payload["planned_hand_joints_dense"]).shape[0] // max(control_steps, 1)), 1)
    rng = np.random.default_rng(int(args.seed))
    config = BagatelleConfig(
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        seed=0,
        control_timestep=0.05,
    )
    rows: list[dict[str, Any]] = []
    changed = 0
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        for i, frame in enumerate(waypoint_frames.tolist()):
            if frame < 0 or frame >= target_keys.shape[0]:
                continue
            target = target_keys[int(frame)]
            if not bool(np.any(target > float(args.threshold))):
                continue
            base = kin.clip_qpos(waypoint_qpos[i])
            base_activation = kin.activation_for_qpos(base, settle_steps=1)
            base_score, base_tp, base_fp, base_fn = contact_counts(target, base_activation, args.threshold)
            indices = active_joint_indices(assignments[i])
            if indices.size == 0:
                continue
            best = base
            best_score, best_tp, best_fp, best_fn = base_score, base_tp, base_fp, base_fn
            best_source = "baseline"
            for sample in range(max(int(args.samples_per_waypoint), 0)):
                candidate = base.copy()
                local_sigma = float(args.sigma) * (1.0 if sample < args.samples_per_waypoint * 0.75 else 0.5)
                candidate[indices] += rng.normal(0.0, local_sigma, size=indices.size).astype(np.float32)
                candidate = kin.clip_qpos(candidate)
                activation = kin.activation_for_qpos(candidate, settle_steps=1)
                score, tp, fp, fn = contact_counts(target, activation, args.threshold)
                if score < base_score + float(args.min_static_improvement):
                    continue
                if fp - base_fp > int(args.max_fp_increase):
                    continue
                rank = (score, tp, -fp, -fn)
                best_rank = (best_score, best_tp, -best_fp, -best_fn)
                if rank > best_rank:
                    best = candidate
                    best_score, best_tp, best_fp, best_fn = score, tp, fp, fn
                    best_source = f"sample_{sample}"
            if best_source != "baseline":
                waypoint_qpos[i] = best.astype(np.float32)
                changed += 1
            if i < 20 or best_source != "baseline":
                rows.append(
                    {
                        "waypoint": int(i),
                        "frame": int(frame),
                        "source": best_source,
                        "baseline_static_f1": float(base_score),
                        "baseline_tp": int(base_tp),
                        "baseline_fp": int(base_fp),
                        "baseline_fn": int(base_fn),
                        "static_f1": float(best_score),
                        "tp": int(best_tp),
                        "fp": int(best_fp),
                        "fn": int(best_fn),
                        "active_joint_count": int(indices.size),
                    }
                )

    control_x = np.arange(control_steps, dtype=np.float64)
    dense_x = np.arange(control_steps * substeps, dtype=np.float64) / float(substeps)
    control_qpos = interp_rows(waypoint_frames, waypoint_qpos, control_x)
    dense_qpos = interp_rows(control_x, control_qpos, dense_x)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    payload["waypoint_hand_joints"] = waypoint_qpos.astype(np.float32)
    payload["planned_hand_joints"] = control_qpos.astype(np.float32)
    payload["planned_hand_joints_dense"] = dense_qpos.astype(np.float32)
    atomic_save_npz(out / "trajectory.npz", **payload)
    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(args.environment_name),
        demo_id=0,
        actions=np.zeros((dense_qpos.shape[0], 39), dtype=np.float32),
        goals=dense_goals,
        hand_joints=dense_qpos,
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
        "changed_waypoints": int(changed),
        "waypoints": int(waypoint_qpos.shape[0]),
        "samples_per_waypoint": int(args.samples_per_waypoint),
        "sigma": float(args.sigma),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "search_rows": rows[:300],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
