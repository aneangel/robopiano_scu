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


FOREARM_INDICES = np.asarray([21, 22, 44, 45], dtype=np.int64)


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def chunks_for_onset(keys: np.ndarray, *, split_key: int, max_keys_per_frame: int) -> list[np.ndarray]:
    values = np.asarray(keys, dtype=np.int32).reshape(-1)
    left = np.sort(values[values < int(split_key)])
    right = np.sort(values[values >= int(split_key)])
    ordered: list[int] = []
    while left.size or right.size:
        if left.size:
            ordered.append(int(left[0]))
            left = left[1:]
        if right.size:
            ordered.append(int(right[0]))
            right = right[1:]
    width = max(int(max_keys_per_frame), 1)
    return [np.asarray(ordered[index : index + width], dtype=np.int32) for index in range(0, len(ordered), width)]


def target_for_keys(keys: np.ndarray) -> np.ndarray:
    out = np.zeros((88,), dtype=np.float32)
    valid = np.asarray(keys, dtype=np.int32)
    valid = valid[(valid >= 0) & (valid < 88)]
    out[valid.astype(np.int64)] = 1.0
    return out


def counts(target: np.ndarray, activation: np.ndarray, threshold: float) -> tuple[int, int, int]:
    goal = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    played = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(goal, played).sum())
    fp = int(np.logical_and(~goal, played).sum())
    fn = int(np.logical_and(goal, ~played).sum())
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * int(tp) + int(fp) + int(fn)
    return float(2 * int(tp) / denom) if denom else 1.0


def rank(target: np.ndarray, activation: np.ndarray, threshold: float, fp_weight: float) -> tuple[float, int, int, int]:
    tp, fp, fn = counts(target, activation, threshold)
    return (f1_from_counts(tp, fp, fn) + 0.10 * tp - float(fp_weight) * fp - 0.03 * fn, tp, -fp, -fn)


def perturb(base: np.ndarray, rng: np.random.Generator, *, finger_sigma: float, forearm_sigma: float) -> np.ndarray:
    out = np.asarray(base, dtype=np.float32).copy()
    out[ALL_FINGER_JOINT_INDICES] += rng.normal(0.0, float(finger_sigma), size=ALL_FINGER_JOINT_INDICES.size).astype(np.float32)
    out[FOREARM_INDICES] += rng.normal(0.0, float(forearm_sigma), size=FOREARM_INDICES.size).astype(np.float32)
    return out


def best_chunk_pose(
    *,
    kin: BagatelleKinematics,
    target: np.ndarray,
    seed_qpos: np.ndarray,
    rng: np.random.Generator,
    threshold: float,
    samples: int,
    iterations: int,
    elite_count: int,
    finger_sigma: float,
    forearm_sigma: float,
    fp_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = kin.clip_qpos(seed_qpos)
    candidates: list[tuple[np.ndarray, np.ndarray, str]] = [(base, kin.activation_for_qpos(base, settle_steps=1)[:88], "seed")]
    centers = [base]
    fs = float(finger_sigma)
    forearm_s = float(forearm_sigma)
    for iteration in range(max(int(iterations), 1)):
        for sample in range(max(int(samples), 0)):
            center = centers[int(sample % len(centers))]
            qpos = kin.clip_qpos(perturb(center, rng, finger_sigma=fs, forearm_sigma=forearm_s))
            activation = kin.activation_for_qpos(qpos, settle_steps=1)[:88]
            candidates.append((qpos, activation, f"iter{iteration}_sample{sample}"))
        candidates.sort(key=lambda item: rank(target, item[1], threshold, fp_weight), reverse=True)
        centers = [row[0] for row in candidates[: max(int(elite_count), 1)]]
        fs *= 0.55
        forearm_s *= 0.55
    candidates.sort(key=lambda item: rank(target, item[1], threshold, fp_weight), reverse=True)
    best_qpos, best_activation, best_source = candidates[0]
    tp, fp, fn = counts(target, best_activation, threshold)
    return best_qpos.astype(np.float32), {
        "source": str(best_source),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "static_f1": float(f1_from_counts(tp, fp, fn)),
        "played_keys": np.flatnonzero(best_activation > float(threshold)).astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prelude-lite: replace high-polyphony onset frames with sub-chord poses.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-key", type=int, default=44)
    parser.add_argument("--max-keys-per-frame", type=int, default=2)
    parser.add_argument("--max-shift-frames", type=int, default=3)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--finger-sigma", type=float, default=0.050)
    parser.add_argument("--forearm-sigma", type=float, default=0.030)
    parser.add_argument("--fp-weight", type=float, default=0.35)
    parser.add_argument("--max-edits", type=int, default=80)
    parser.add_argument("--seed", type=int, default=311)
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    active = target_keys > float(args.threshold)
    control = np.asarray(payload["planned_hand_joints"], dtype=np.float32).copy()
    original_control = control.copy()
    dense_base = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32)
    substeps = max(int(dense_base.shape[0] // max(control.shape[0], 1)), 1)
    previous = np.zeros((88,), dtype=bool)
    edits: list[tuple[int, np.ndarray]] = []
    for frame, row in enumerate(active):
        onsets = np.flatnonzero(row & ~previous).astype(np.int32)
        if onsets.size > max(int(args.max_keys_per_frame), 1):
            for chunk_index, chunk in enumerate(
                chunks_for_onset(onsets, split_key=int(args.split_key), max_keys_per_frame=int(args.max_keys_per_frame))
            ):
                target_frame = min(int(frame) + min(int(chunk_index), int(args.max_shift_frames)), control.shape[0] - 1)
                edits.append((target_frame, chunk))
        previous = row
    edits = edits[: max(int(args.max_edits), 0)]

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    config = BagatelleConfig(environment_name=str(args.environment_name), threshold=float(args.threshold), seed=0)
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        for edit_index, (frame, chunk) in enumerate(edits):
            qpos, meta = best_chunk_pose(
                kin=kin,
                target=target_for_keys(chunk),
                seed_qpos=control[int(frame)],
                rng=rng,
                threshold=float(args.threshold),
                samples=int(args.samples),
                iterations=int(args.iterations),
                elite_count=int(args.elite_count),
                finger_sigma=float(args.finger_sigma),
                forearm_sigma=float(args.forearm_sigma),
                fp_weight=float(args.fp_weight),
            )
            control[int(frame)] = qpos
            rows.append({"edit": int(edit_index), "frame": int(frame), "keys": chunk.astype(int).tolist(), **meta})

    dense = np.repeat(control, substeps, axis=0).astype(np.float32)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    payload["planned_hand_joints"] = control.astype(np.float32)
    payload["planned_hand_joints_dense"] = dense
    payload["prelude_lite_changed_frames"] = (np.linalg.norm(control - original_control, axis=1) > 1e-6).astype(np.float32)
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
        "edits": int(len(edits)),
        "changed_control_frames": int(np.count_nonzero(np.linalg.norm(control - original_control, axis=1) > 1e-6)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "edit_rows": rows[:200],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
