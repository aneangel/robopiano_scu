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


def onset_frames(target_keys: np.ndarray, threshold: float) -> list[int]:
    active = np.asarray(target_keys, dtype=np.float32)[:, :88] > float(threshold)
    previous = np.zeros((88,), dtype=bool)
    out: list[int] = []
    for frame, row in enumerate(active):
        if bool(np.any(row & ~previous)):
            out.append(int(frame))
        previous = row
    return out


def contact_counts(target: np.ndarray, activation: np.ndarray, threshold: float) -> tuple[float, int, int, int]:
    goal = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    played = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(goal, played).sum())
    fp = int(np.logical_and(~goal, played).sum())
    fn = int(np.logical_and(goal, ~played).sum())
    denom = 2 * tp + fp + fn
    return (float(2 * tp / denom) if denom else 1.0, tp, fp, fn)


def rank_score(score: tuple[float, int, int, int], *, fp_weight: float, fn_weight: float) -> tuple[float, int, int, int]:
    f1, tp, fp, fn = score
    return (float(f1) + 0.04 * int(tp) - float(fp_weight) * int(fp) - float(fn_weight) * int(fn), int(tp), -int(fp), -int(fn))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tarantella: one contact-validated pulse per target onset mask.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pulse-width", type=int, default=2)
    parser.add_argument("--lead-substeps", type=int, default=0)
    parser.add_argument("--min-static-improvement", type=float, default=0.20)
    parser.add_argument("--max-static-fp", type=int, default=1)
    parser.add_argument("--fp-rank-weight", type=float, default=0.35)
    parser.add_argument("--fn-rank-weight", type=float, default=0.08)
    parser.add_argument("--settle-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control = np.asarray(payload["planned_hand_joints"], dtype=np.float32)
    dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32).copy()
    substeps = max(int(dense.shape[0] // max(control.shape[0], 1)), 1)
    dense_dt = 0.05 / float(substeps)

    config = BagatelleConfig(
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        seed=int(args.seed),
        control_timestep=0.05,
    )
    injected = np.zeros((dense.shape[0],), dtype=bool)
    rows: list[dict[str, Any]] = []
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        activations = np.stack(
            [
                kin.activation_for_qpos(qpos, settle_steps=max(int(args.settle_steps), 1))[:88].astype(np.float32)
                for qpos in control
            ],
            axis=0,
        )
        for frame in onset_frames(target_keys, float(args.threshold)):
            target = target_keys[frame]
            base_score = contact_counts(target, activations[frame], float(args.threshold))
            best_score = base_score
            best_frame = int(frame)
            best_qpos = control[frame]
            for candidate_frame, activation in enumerate(activations):
                score = contact_counts(target, activation, float(args.threshold))
                if score[2] > int(args.max_static_fp):
                    continue
                if score[0] < base_score[0] + float(args.min_static_improvement):
                    continue
                if rank_score(score, fp_weight=float(args.fp_rank_weight), fn_weight=float(args.fn_rank_weight)) > rank_score(
                    best_score,
                    fp_weight=float(args.fp_rank_weight),
                    fn_weight=float(args.fn_rank_weight),
                ):
                    best_score = score
                    best_frame = int(candidate_frame)
                    best_qpos = control[candidate_frame]
            use = best_frame != int(frame)
            if use:
                start = int(frame) * substeps - max(int(args.lead_substeps), 0)
                start = max(start, 0)
                end = min(start + max(int(args.pulse_width), 1), dense.shape[0])
                if start < end:
                    dense[start:end] = best_qpos.astype(np.float32)
                    injected[start:end] = True
            rows.append(
                {
                    "target_frame": int(frame),
                    "use": bool(use),
                    "candidate_frame": int(best_frame),
                    "base_static_f1": float(base_score[0]),
                    "base_tp": int(base_score[1]),
                    "base_fp": int(base_score[2]),
                    "base_fn": int(base_score[3]),
                    "best_static_f1": float(best_score[0]),
                    "best_tp": int(best_score[1]),
                    "best_fp": int(best_score[2]),
                    "best_fn": int(best_score[3]),
                }
            )

    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["tarantella_injected_dense_frames"] = injected.astype(np.float32)
    atomic_save_npz(out / "trajectory.npz", **payload)

    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
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
        "pulse_width": int(args.pulse_width),
        "lead_substeps": int(args.lead_substeps),
        "min_static_improvement": float(args.min_static_improvement),
        "max_static_fp": int(args.max_static_fp),
        "onsets": int(len(rows)),
        "used_onsets": int(sum(1 for row in rows if row["use"])),
        "injected_dense_frames": int(np.count_nonzero(injected)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "onset_rows": rows[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
