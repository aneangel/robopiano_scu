#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

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
from impromptu.joint_space_trajectory import BAGATELLE_FINGER_JOINT_INDEX_ROWS  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def mask_key(row: np.ndarray, threshold: float) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(np.asarray(row, dtype=np.float32)[:88] > float(threshold)).tolist())


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


def assigned_target_fingers(kin: BagatelleKinematics, qpos: np.ndarray, keys: tuple[int, ...]) -> np.ndarray:
    if not keys:
        return np.zeros((0,), dtype=np.int64)
    fingertips = np.asarray(kin.fingertip_positions_for_qpos(qpos), dtype=np.float32)
    targets = np.asarray(kin.key_press_targets(np.asarray(keys, dtype=np.int32)), dtype=np.float32)
    cost = np.linalg.norm(fingertips[:, None, :] - targets[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    order = np.argsort(cols, kind="stable")
    return rows[order].astype(np.int64)


def inactive_joint_indices(active_fingers: np.ndarray) -> np.ndarray:
    active = {int(finger) for finger in np.asarray(active_fingers, dtype=np.int64).reshape(-1)}
    rows: list[int] = []
    for finger, joint_indices in enumerate(BAGATELLE_FINGER_JOINT_INDEX_ROWS):
        if finger not in active:
            rows.extend(int(index) for index in joint_indices)
    return np.unique(np.asarray(rows, dtype=np.int64)).astype(np.int64) if rows else np.zeros((0,), dtype=np.int64)


def isolate_candidate(
    *,
    base: np.ndarray,
    replacement: np.ndarray,
    indices: np.ndarray,
    alpha: float,
    kin: BagatelleKinematics,
) -> np.ndarray:
    out = np.asarray(base, dtype=np.float32).copy()
    if indices.size:
        out[indices] = ((1.0 - float(alpha)) * out[indices] + float(alpha) * replacement[indices]).astype(np.float32)
    return kin.clip_qpos(out)


def rank(score: tuple[float, int, int, int], *, fp_weight: float, fn_weight: float) -> tuple[float, int, int, int]:
    f1, tp, fp, fn = score
    return (float(f1) + 0.04 * int(tp) - float(fp_weight) * int(fp) - float(fn_weight) * int(fn), int(tp), -int(fp), -int(fn))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bourree: static contact repair by isolating inactive fingers.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-mean-static-improvement", type=float, default=0.02)
    parser.add_argument("--max-fp-increase", type=float, default=0.0)
    parser.add_argument("--fp-rank-weight", type=float, default=0.18)
    parser.add_argument("--fn-rank-weight", type=float, default=0.04)
    parser.add_argument("--settle-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control = np.asarray(payload["planned_hand_joints"], dtype=np.float32).copy()
    dense_base = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32)
    substeps = max(int(dense_base.shape[0] // max(control.shape[0], 1)), 1)

    masks: dict[tuple[int, ...], list[int]] = {}
    for frame, row in enumerate(target_keys):
        key = mask_key(row, float(args.threshold))
        if key:
            masks.setdefault(key, []).append(int(frame))

    bag_cfg = BagatelleConfig(
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        seed=int(args.seed),
        control_timestep=0.05,
    )
    selected: dict[tuple[int, ...], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    with BagatelleKinematics(config=bag_cfg, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        neutral = np.asarray(kin.neutral_qpos, dtype=np.float32)
        straight = kin.clip_qpos(np.zeros_like(neutral, dtype=np.float32))
        replacements = (("neutral", neutral), ("straight", straight))
        for key, frames in sorted(masks.items(), key=lambda item: (len(item[0]), item[0])):
            target = target_keys[frames[0]]
            baseline_scores: list[tuple[float, int, int, int]] = []
            seed_rows: list[tuple[tuple[float, int, int, int], np.ndarray, int]] = []
            for frame in frames:
                activation = kin.activation_for_qpos(control[frame], settle_steps=max(int(args.settle_steps), 1))
                score = contact_counts(target, activation, float(args.threshold))
                baseline_scores.append(score)
                seed_rows.append((score, control[frame].copy(), int(frame)))
            base_mean_f1 = float(np.mean([score[0] for score in baseline_scores]))
            base_mean_fp = float(np.mean([score[2] for score in baseline_scores]))
            seed_rows.sort(
                key=lambda item: rank(item[0], fp_weight=float(args.fp_rank_weight), fn_weight=float(args.fn_rank_weight)),
                reverse=True,
            )
            best_score, best_qpos, best_frame = seed_rows[0]
            best_source = f"baseline_frame_{best_frame}"
            active = assigned_target_fingers(kin, best_qpos, key)
            indices = inactive_joint_indices(active)
            candidates: list[tuple[tuple[float, int, int, int], np.ndarray, str]] = [(best_score, best_qpos, best_source)]
            for replacement_name, replacement in replacements:
                for alpha in (0.25, 0.50, 0.75, 1.0):
                    cand = isolate_candidate(
                        base=best_qpos,
                        replacement=replacement,
                        indices=indices,
                        alpha=float(alpha),
                        kin=kin,
                    )
                    activation = kin.activation_for_qpos(cand, settle_steps=max(int(args.settle_steps), 1))
                    candidates.append(
                        (
                            contact_counts(target, activation, float(args.threshold)),
                            cand,
                            f"{replacement_name}_inactive_alpha_{alpha:.2f}",
                        )
                    )
            candidates.sort(
                key=lambda item: rank(item[0], fp_weight=float(args.fp_rank_weight), fn_weight=float(args.fn_rank_weight)),
                reverse=True,
            )
            score, qpos, source_name = candidates[0]
            use = (
                float(score[0]) >= base_mean_f1 + float(args.min_mean_static_improvement)
                and float(score[2]) <= base_mean_fp + float(args.max_fp_increase)
                and source_name != best_source
            )
            if use:
                selected[key] = qpos.astype(np.float32)
            rows.append(
                {
                    "keys": list(key),
                    "frames": int(len(frames)),
                    "use": bool(use),
                    "source": source_name,
                    "active_fingers": active.astype(int).tolist(),
                    "baseline_mean_static_f1": base_mean_f1,
                    "baseline_mean_fp": base_mean_fp,
                    "best_static_f1": float(score[0]),
                    "best_tp": int(score[1]),
                    "best_fp": int(score[2]),
                    "best_fn": int(score[3]),
                }
            )

    changed = 0
    for key, frames in masks.items():
        qpos = selected.get(key)
        if qpos is None:
            continue
        for frame in frames:
            control[frame] = qpos
            changed += 1
    dense = np.repeat(control, substeps, axis=0).astype(np.float32)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    payload["planned_hand_joints"] = control.astype(np.float32)
    payload["planned_hand_joints_dense"] = dense
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
        "masks": int(len(masks)),
        "used_masks": int(len(selected)),
        "changed_control_frames": int(changed),
        "control_steps": int(control.shape[0]),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "isolation_rows": rows[:300],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
