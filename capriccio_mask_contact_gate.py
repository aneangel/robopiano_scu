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
from impromptu.joint_space_trajectory import ALL_FINGER_JOINT_INDICES  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def mask_key(row: np.ndarray, threshold: float) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(np.asarray(row, dtype=np.float32)[:88] > float(threshold)).tolist())


def contact_score(target: np.ndarray, activation: np.ndarray, threshold: float) -> tuple[float, int, int, int]:
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
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask-level static contact optimizer for Impromptu control hand states.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--samples-per-mask", type=int, default=80)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--top-seeds", type=int, default=4)
    parser.add_argument("--min-mean-static-improvement", type=float, default=0.05)
    parser.add_argument("--max-fp-increase", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=524)
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control_qpos = np.asarray(payload["planned_hand_joints"], dtype=np.float32).copy()
    dense_base = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32)
    substeps = max(int(dense_base.shape[0] // max(control_qpos.shape[0], 1)), 1)
    rng = np.random.default_rng(int(args.seed))
    masks: dict[tuple[int, ...], list[int]] = {}
    for frame, row in enumerate(target_keys):
        key = mask_key(row, args.threshold)
        if key:
            masks.setdefault(key, []).append(int(frame))
    config = BagatelleConfig(environment_name=str(args.environment_name), threshold=float(args.threshold), seed=0)
    selected_by_mask: dict[tuple[int, ...], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        for key, frames in sorted(masks.items(), key=lambda item: (len(item[0]), item[0])):
            target = target_keys[frames[0]]
            seed_scores: list[tuple[tuple[float, int, int, int], np.ndarray, int]] = []
            baseline_scores = []
            for frame in frames:
                activation = kin.activation_for_qpos(control_qpos[frame], settle_steps=1)
                score = contact_score(target, activation, args.threshold)
                baseline_scores.append(score)
                seed_scores.append((score, control_qpos[frame].copy(), int(frame)))
            seed_scores.sort(key=lambda item: (item[0][0], item[0][1], -item[0][2], -item[0][3]), reverse=True)
            base_mean_f1 = float(np.mean([s[0] for s in baseline_scores]))
            base_mean_fp = float(np.mean([s[2] for s in baseline_scores]))
            candidates: list[tuple[tuple[float, int, int, int], np.ndarray, str]] = [
                (score, qpos, f"frame_{frame}") for score, qpos, frame in seed_scores[: max(int(args.top_seeds), 1)]
            ]
            top_qpos = [q for _score, q, _frame in seed_scores[: max(int(args.top_seeds), 1)]]
            for sample in range(max(int(args.samples_per_mask), 0)):
                base = top_qpos[int(sample % len(top_qpos))]
                cand = base.copy()
                cand[ALL_FINGER_JOINT_INDICES] += rng.normal(
                    0.0, float(args.sigma), size=ALL_FINGER_JOINT_INDICES.size
                ).astype(np.float32)
                cand = kin.clip_qpos(cand)
                activation = kin.activation_for_qpos(cand, settle_steps=1)
                candidates.append((contact_score(target, activation, args.threshold), cand, f"sample_{sample}"))
            candidates.sort(key=lambda item: (item[0][0], item[0][1], -item[0][2], -item[0][3]), reverse=True)
            best_score, best_qpos, best_source = candidates[0]
            use = (
                float(best_score[0]) >= base_mean_f1 + float(args.min_mean_static_improvement)
                and float(best_score[2]) <= base_mean_fp + float(args.max_fp_increase)
            )
            if use:
                selected_by_mask[key] = best_qpos.astype(np.float32)
            rows.append(
                {
                    "keys": list(key),
                    "frames": int(len(frames)),
                    "use": bool(use),
                    "source": best_source,
                    "baseline_mean_static_f1": base_mean_f1,
                    "baseline_mean_fp": base_mean_fp,
                    "best_static_f1": float(best_score[0]),
                    "best_tp": int(best_score[1]),
                    "best_fp": int(best_score[2]),
                    "best_fn": int(best_score[3]),
                }
            )
    changed_frames = 0
    for key, frames in masks.items():
        if key not in selected_by_mask:
            continue
        for frame in frames:
            control_qpos[frame] = selected_by_mask[key]
            changed_frames += 1
    dense_qpos = np.repeat(control_qpos, substeps, axis=0).astype(np.float32)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    payload["planned_hand_joints"] = control_qpos.astype(np.float32)
    payload["planned_hand_joints_dense"] = dense_qpos
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
        "masks": int(len(masks)),
        "used_masks": int(len(selected_by_mask)),
        "changed_control_frames": int(changed_frames),
        "control_steps": int(control_qpos.shape[0]),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "mask_rows": rows[:300],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
