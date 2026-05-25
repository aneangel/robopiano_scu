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


def target_onsets(target: np.ndarray, threshold: float) -> list[tuple[int, list[int]]]:
    active = np.asarray(target, dtype=np.float32)[:, :88] > float(threshold)
    previous = np.zeros((88,), dtype=bool)
    rows: list[tuple[int, list[int]]] = []
    for frame, row in enumerate(active):
        keys = np.flatnonzero(row & ~previous).astype(int).tolist()
        if keys:
            rows.append((int(frame), keys))
        previous = row
    return rows


def frame_contact_counts(target: np.ndarray, activation: np.ndarray, threshold: float) -> tuple[float, int, int, int]:
    goal = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    played = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(goal, played).sum())
    fp = int(np.logical_and(~goal, played).sum())
    fn = int(np.logical_and(goal, ~played).sum())
    denom = 2 * tp + fp + fn
    return (float(2 * tp / denom) if denom else 1.0, tp, fp, fn)


def build_activation_library(
    *,
    kin: BagatelleKinematics,
    qpos_rows: np.ndarray,
    target_keys: np.ndarray,
    threshold: float,
    settle_steps: int,
) -> tuple[np.ndarray, dict[int, list[dict[str, Any]]]]:
    activations: list[np.ndarray] = []
    library: dict[int, list[dict[str, Any]]] = {key: [] for key in range(88)}
    for frame, qpos in enumerate(np.asarray(qpos_rows, dtype=np.float32)):
        activation = kin.activation_for_qpos(qpos, settle_steps=max(int(settle_steps), 1))[:88]
        activations.append(activation.astype(np.float32))
        played = np.flatnonzero(activation > float(threshold)).astype(int).tolist()
        played_count = int(len(played))
        for key in played:
            library[int(key)].append(
                {
                    "frame": int(frame),
                    "qpos": np.asarray(qpos, dtype=np.float32).copy(),
                    "activation": activation.astype(np.float32),
                    "played_count": played_count,
                }
            )
    for key, rows in library.items():
        rows.sort(key=lambda row: (int(row["played_count"]), int(row["frame"])))
    return np.stack(activations, axis=0).astype(np.float32), library


def choose_candidate(
    *,
    key: int,
    target_row: np.ndarray,
    rows: list[dict[str, Any]],
    threshold: float,
    max_static_fp: int,
    min_static_f1: float,
    max_candidate_played_count: int,
) -> dict[str, Any] | None:
    best: tuple[tuple[float, int, int, int, int], dict[str, Any]] | None = None
    for row in rows:
        if int(max_candidate_played_count) > 0 and int(row["played_count"]) > int(max_candidate_played_count):
            continue
        score, tp, fp, fn = frame_contact_counts(target_row, row["activation"], threshold)
        if fp > int(max_static_fp):
            continue
        if score < float(min_static_f1):
            continue
        rank = (float(score), int(tp), -int(fp), -int(fn), -int(row["played_count"]))
        if best is None or rank > best[0]:
            best = (rank, row)
    return None if best is None else best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cadenza: inject short dense pulses for target onsets missed by static contact.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pulse-width", type=int, default=2)
    parser.add_argument("--pulse-gap", type=int, default=0)
    parser.add_argument("--lead-substeps", type=int, default=0)
    parser.add_argument("--max-pulses-per-onset", type=int, default=10)
    parser.add_argument("--max-static-fp", type=int, default=2)
    parser.add_argument("--min-static-f1", type=float, default=0.25)
    parser.add_argument("--max-candidate-played-count", type=int, default=0)
    parser.add_argument("--only-static-missed", action="store_true")
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

    bag_cfg = BagatelleConfig(
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        seed=int(args.seed),
        control_timestep=0.05,
    )
    injected = np.zeros((dense.shape[0],), dtype=bool)
    injections: list[dict[str, Any]] = []
    with BagatelleKinematics(config=bag_cfg, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        static_activations, library = build_activation_library(
            kin=kin,
            qpos_rows=control,
            target_keys=target_keys,
            threshold=float(args.threshold),
            settle_steps=int(args.settle_steps),
        )
        for frame, keys in target_onsets(target_keys, float(args.threshold)):
            target_row = target_keys[frame]
            offset = -max(int(args.lead_substeps), 0)
            pulses_used = 0
            for key in keys:
                if bool(args.only_static_missed) and static_activations[frame, int(key)] > float(args.threshold):
                    continue
                candidate = choose_candidate(
                    key=int(key),
                    target_row=target_row,
                    rows=library.get(int(key), []),
                    threshold=float(args.threshold),
                    max_static_fp=int(args.max_static_fp),
                    min_static_f1=float(args.min_static_f1),
                    max_candidate_played_count=int(args.max_candidate_played_count),
                )
                if candidate is None:
                    continue
                start = int(frame) * substeps + int(offset)
                width = max(int(args.pulse_width), 1)
                end = start + width
                start = max(start, 0)
                end = min(end, dense.shape[0])
                if start >= end:
                    continue
                dense[start:end] = np.asarray(candidate["qpos"], dtype=np.float32)
                injected[start:end] = True
                injections.append(
                    {
                        "target_frame": int(frame),
                        "target_key": int(key),
                        "dense_start": int(start),
                        "dense_end": int(end),
                        "candidate_frame": int(candidate["frame"]),
                        "candidate_played_count": int(candidate["played_count"]),
                    }
                )
                offset += width + max(int(args.pulse_gap), 0)
                pulses_used += 1
                if pulses_used >= max(int(args.max_pulses_per_onset), 1):
                    break

    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["cadenza_injected_dense_frames"] = injected.astype(np.float32)
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
        "lead_substeps": int(args.lead_substeps),
        "max_static_fp": int(args.max_static_fp),
        "min_static_f1": float(args.min_static_f1),
        "max_candidate_played_count": int(args.max_candidate_played_count),
        "only_static_missed": bool(args.only_static_missed),
        "injections": int(len(injections)),
        "injected_dense_frames": int(np.count_nonzero(injected)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "injection_rows": injections[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
