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


def onset_rows(target_keys: np.ndarray, threshold: float) -> list[tuple[int, np.ndarray]]:
    active = np.asarray(target_keys, dtype=np.float32)[:, :88] > float(threshold)
    previous = np.zeros((88,), dtype=bool)
    out: list[tuple[int, np.ndarray]] = []
    for frame, row in enumerate(active):
        onset = np.flatnonzero(row & ~previous).astype(np.int32)
        if onset.size:
            out.append((int(frame), onset))
        previous = row
    return out


def frame_f1(target: np.ndarray, played: np.ndarray, threshold: float) -> tuple[float, int, int, int]:
    goal = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    pred = np.asarray(played, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(goal, pred).sum())
    fp = int(np.logical_and(~goal, pred).sum())
    fn = int(np.logical_and(goal, ~pred).sum())
    denom = 2 * tp + fp + fn
    return (float(2 * tp / denom) if denom else 1.0, tp, fp, fn)


def candidate_score(
    *,
    target_row: np.ndarray,
    missing_keys: np.ndarray,
    activation: np.ndarray,
    threshold: float,
    max_static_fp: int,
    max_candidate_played_count: int,
    hit_weight: float,
    fp_weight: float,
    played_weight: float,
) -> tuple[float, dict[str, Any]]:
    played = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
    target = np.asarray(target_row, dtype=np.float32)[:88] > float(threshold)
    missing = np.asarray(missing_keys, dtype=np.int32).reshape(-1)
    hits = np.intersect1d(np.flatnonzero(played).astype(np.int32), missing, assume_unique=False)
    fp = int(np.logical_and(~target, played).sum())
    played_count = int(np.count_nonzero(played))
    f1, tp, _fp, fn = frame_f1(target_row, activation, threshold)
    if fp > int(max_static_fp):
        return -np.inf, {}
    if int(max_candidate_played_count) > 0 and played_count > int(max_candidate_played_count):
        return -np.inf, {}
    value = (
        float(hit_weight) * float(hits.size)
        + 0.10 * float(tp)
        + 0.05 * float(f1)
        - float(fp_weight) * float(fp)
        - float(played_weight) * float(played_count)
        - 0.03 * float(fn)
    )
    return float(value), {
        "hits": hits.astype(int).tolist(),
        "fp": int(fp),
        "tp": int(tp),
        "fn": int(fn),
        "played_count": int(played_count),
        "static_f1": float(f1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Siciliana: grouped set-cover event injection for missed onset keys.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pulse-width", type=int, default=1)
    parser.add_argument("--pulse-gap", type=int, default=0)
    parser.add_argument("--lead-substeps", type=int, default=0)
    parser.add_argument("--max-pulses-per-onset", type=int, default=2)
    parser.add_argument("--max-static-fp", type=int, default=0)
    parser.add_argument("--max-candidate-played-count", type=int, default=0)
    parser.add_argument("--rest-source", choices=("baseline", "neutral"), default="baseline")
    parser.add_argument("--hit-weight", type=float, default=1.0)
    parser.add_argument("--fp-weight", type=float, default=2.0)
    parser.add_argument("--played-weight", type=float, default=0.05)
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
    injection_rows: list[dict[str, Any]] = []
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        if str(args.rest_source) == "neutral":
            dense = np.repeat(np.asarray(kin.neutral_qpos, dtype=np.float32).reshape(1, -1), dense.shape[0], axis=0)
        activations = np.stack(
            [
                kin.activation_for_qpos(qpos, settle_steps=max(int(args.settle_steps), 1))[:88].astype(np.float32)
                for qpos in control
            ],
            axis=0,
        )
        for frame, onset_keys in onset_rows(target_keys, float(args.threshold)):
            static_played = activations[frame] > float(args.threshold)
            remaining = np.asarray([key for key in onset_keys.tolist() if not bool(static_played[int(key)])], dtype=np.int32)
            if remaining.size == 0:
                continue
            selected: list[tuple[int, dict[str, Any]]] = []
            used_frames: set[int] = set()
            for _ in range(max(int(args.max_pulses_per_onset), 1)):
                best_value = -np.inf
                best_frame = -1
                best_meta: dict[str, Any] = {}
                for cand_frame, activation in enumerate(activations):
                    if int(cand_frame) in used_frames:
                        continue
                    value, meta = candidate_score(
                        target_row=target_keys[frame],
                        missing_keys=remaining,
                        activation=activation,
                        threshold=float(args.threshold),
                        max_static_fp=int(args.max_static_fp),
                        max_candidate_played_count=int(args.max_candidate_played_count),
                        hit_weight=float(args.hit_weight),
                        fp_weight=float(args.fp_weight),
                        played_weight=float(args.played_weight),
                    )
                    if value > best_value:
                        best_value = float(value)
                        best_frame = int(cand_frame)
                        best_meta = meta
                if best_frame < 0 or not best_meta.get("hits"):
                    break
                selected.append((best_frame, best_meta))
                used_frames.add(best_frame)
                covered = set(int(key) for key in best_meta["hits"])
                remaining = np.asarray([int(key) for key in remaining.tolist() if int(key) not in covered], dtype=np.int32)
                if remaining.size == 0:
                    break
            offset = -max(int(args.lead_substeps), 0)
            for cand_frame, meta in selected:
                start = int(frame) * substeps + int(offset)
                start = max(start, 0)
                end = min(start + max(int(args.pulse_width), 1), dense.shape[0])
                if start >= end:
                    continue
                dense[start:end] = control[cand_frame].astype(np.float32)
                injected[start:end] = True
                injection_rows.append(
                    {
                        "target_frame": int(frame),
                        "candidate_frame": int(cand_frame),
                        "dense_start": int(start),
                        "dense_end": int(end),
                        **meta,
                    }
                )
                offset += max(int(args.pulse_width), 1) + max(int(args.pulse_gap), 0)

    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["siciliana_injected_dense_frames"] = injected.astype(np.float32)
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
        "pulse_gap": int(args.pulse_gap),
        "max_pulses_per_onset": int(args.max_pulses_per_onset),
        "max_static_fp": int(args.max_static_fp),
        "max_candidate_played_count": int(args.max_candidate_played_count),
        "rest_source": str(args.rest_source),
        "injections": int(len(injection_rows)),
        "injected_dense_frames": int(np.count_nonzero(injected)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "injection_rows": injection_rows[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
