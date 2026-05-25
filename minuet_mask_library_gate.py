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
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def mask_key(row: np.ndarray, threshold: float) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(np.asarray(row, dtype=np.float32)[:88] > float(threshold)).tolist())


def contact_counts(target: np.ndarray, activation: np.ndarray, threshold: float) -> tuple[int, int, int]:
    goal = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    played = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(goal, played).sum())
    fp = int(np.logical_and(~goal, played).sum())
    fn = int(np.logical_and(goal, ~played).sum())
    return tp, fp, fn


def contact_f1(target: np.ndarray, activation: np.ndarray, threshold: float) -> float:
    tp, fp, fn = contact_counts(target, activation, threshold)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 1.0


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def unique_qpos(rows: list[np.ndarray], *, decimals: int = 5) -> np.ndarray:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for row in rows:
        qpos = np.asarray(row, dtype=np.float32).reshape(-1)
        key = np.round(qpos, int(decimals)).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(qpos.astype(np.float32))
    if not out:
        return np.zeros((0, 46), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)


def build_candidate_rows(
    *,
    source: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    include_dense: bool,
    max_dense_rows: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for payload in (source, candidate):
        for key in ("planned_hand_joints", "waypoint_hand_joints", "ik_anchor_qpos"):
            if key in payload:
                rows.extend(np.asarray(payload[key], dtype=np.float32).reshape(-1, 46))
    if include_dense:
        dense = np.asarray(candidate.get("planned_hand_joints_dense", np.zeros((0, 46))), dtype=np.float32).reshape(-1, 46)
        if dense.shape[0] > 0:
            stride = max(int(np.ceil(dense.shape[0] / max(int(max_dense_rows), 1))), 1)
            rows.extend(dense[::stride])
        for mask_key_name in ("siciliana_injected_dense_frames", "cadenza_injected_dense_frames"):
            if mask_key_name in candidate:
                mask = np.asarray(candidate[mask_key_name], dtype=np.float32).reshape(-1) > 0.5
                if mask.shape[0] == dense.shape[0]:
                    rows.extend(dense[mask])
    return unique_qpos(rows)


def choose_qpos_for_mask(
    *,
    target: np.ndarray,
    previous_qpos: np.ndarray,
    candidates: np.ndarray,
    activations: np.ndarray,
    threshold: float,
    max_static_fp: int,
    tp_weight: float,
    fp_weight: float,
    fn_weight: float,
    played_weight: float,
    motion_weight: float,
    min_static_f1: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    best_score = -np.inf
    best_index = -1
    best_meta: dict[str, Any] = {}
    fallback_score = -np.inf
    fallback_index = -1
    fallback_meta: dict[str, Any] = {}
    target_bool = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    for index, activation in enumerate(activations):
        played_bool = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
        tp, fp, fn = contact_counts(target, activation, threshold)
        played_count = int(np.count_nonzero(played_bool))
        f1 = contact_f1(target, activation, threshold)
        onset_hit_bonus = float(np.logical_and(target_bool, played_bool).sum())
        motion = float(np.linalg.norm(candidates[index] - previous_qpos))
        score = (
            float(tp_weight) * float(tp)
            + 0.10 * onset_hit_bonus
            + 0.05 * float(f1)
            - float(fp_weight) * float(fp)
            - float(fn_weight) * float(fn)
            - float(played_weight) * float(played_count)
            - float(motion_weight) * float(motion)
        )
        meta = {
            "candidate_index": int(index),
            "static_f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "played_count": int(played_count),
            "score": float(score),
            "motion": float(motion),
        }
        if score > fallback_score:
            fallback_score = float(score)
            fallback_index = int(index)
            fallback_meta = meta
        if fp > int(max_static_fp) or f1 < float(min_static_f1):
            continue
        if score > best_score:
            best_score = float(score)
            best_index = int(index)
            best_meta = meta
    if best_index < 0:
        best_index = fallback_index
        best_meta = {**fallback_meta, "fallback": True}
    else:
        best_meta = {**best_meta, "fallback": False}
    return candidates[best_index].astype(np.float32), activations[best_index].astype(np.float32), best_meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minuet: replace dense rollout with one contact-validated hand state per active key mask."
    )
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-static-fp", type=int, default=0)
    parser.add_argument("--min-static-f1", type=float, default=0.0)
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--max-dense-rows", type=int, default=400)
    parser.add_argument("--settle-steps", type=int, default=1)
    parser.add_argument("--tp-weight", type=float, default=2.0)
    parser.add_argument("--fp-weight", type=float, default=4.0)
    parser.add_argument("--fn-weight", type=float, default=1.0)
    parser.add_argument("--played-weight", type=float, default=0.05)
    parser.add_argument("--motion-weight", type=float, default=0.02)
    parser.add_argument("--rest-source", choices=("neutral", "source"), default="neutral")
    parser.add_argument("--overlay-candidate-injections", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = load_npz(Path(args.source_npz))
    candidate = load_npz(Path(args.candidate_npz))
    target_keys = np.asarray(candidate["target_keys"], dtype=np.float32)[:, :88]
    base_dense = np.asarray(candidate["planned_hand_joints_dense"], dtype=np.float32)
    source_dense = np.asarray(source["planned_hand_joints_dense"], dtype=np.float32)
    control = np.asarray(candidate["planned_hand_joints"], dtype=np.float32)
    substeps = max(int(base_dense.shape[0] // max(control.shape[0], 1)), 1)
    dense_dt = 0.05 / float(substeps)

    qpos_rows = build_candidate_rows(
        source=source,
        candidate=candidate,
        include_dense=bool(args.include_dense),
        max_dense_rows=int(args.max_dense_rows),
    )
    cfg = BagatelleConfig(
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        seed=int(args.seed),
        control_timestep=0.05,
    )

    report: list[dict[str, Any]] = []
    with BagatelleKinematics(config=cfg, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        if qpos_rows.shape[0] == 0:
            qpos_rows = kin.neutral_qpos.reshape(1, -1).astype(np.float32)
        qpos_rows = np.concatenate([kin.neutral_qpos.reshape(1, -1).astype(np.float32), qpos_rows], axis=0)
        qpos_rows = unique_qpos([row for row in qpos_rows])
        activations = np.stack(
            [
                kin.activation_for_qpos(qpos, settle_steps=max(int(args.settle_steps), 1))[:88].astype(np.float32)
                for qpos in qpos_rows
            ],
            axis=0,
        )
        neutral = np.asarray(kin.neutral_qpos, dtype=np.float32)
        selected_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        previous_qpos = neutral.copy()
        dense = np.zeros_like(base_dense, dtype=np.float32)
        changed_control = 0
        fallback_frames = 0
        for frame, target in enumerate(target_keys):
            key = mask_key(target, float(args.threshold))
            start = int(frame) * substeps
            end = min(start + substeps, dense.shape[0])
            if start >= end:
                continue
            if not key:
                if str(args.rest_source) == "source":
                    dense[start:end] = source_dense[start:end]
                    previous_qpos = source_dense[start].astype(np.float32)
                else:
                    dense[start:end] = neutral
                    previous_qpos = neutral.copy()
                continue
            if key not in selected_cache:
                selected_cache[key] = choose_qpos_for_mask(
                    target=target,
                    previous_qpos=previous_qpos,
                    candidates=qpos_rows,
                    activations=activations,
                    threshold=float(args.threshold),
                    max_static_fp=int(args.max_static_fp),
                    tp_weight=float(args.tp_weight),
                    fp_weight=float(args.fp_weight),
                    fn_weight=float(args.fn_weight),
                    played_weight=float(args.played_weight),
                    motion_weight=float(args.motion_weight),
                    min_static_f1=float(args.min_static_f1),
                )
            qpos, activation, meta = selected_cache[key]
            dense[start:end] = qpos
            previous_qpos = qpos.astype(np.float32)
            if np.linalg.norm(qpos - base_dense[start]) > 1e-6:
                changed_control += 1
            if bool(meta.get("fallback", False)):
                fallback_frames += 1
            if len(report) < 400:
                report.append(
                    {
                        "frame": int(frame),
                        "keys": list(key),
                        **meta,
                    }
                )

    overlay_frames = 0
    if bool(args.overlay_candidate_injections):
        overlay_mask = np.zeros((dense.shape[0],), dtype=bool)
        for mask_key_name in ("siciliana_injected_dense_frames", "cadenza_injected_dense_frames"):
            if mask_key_name in candidate:
                values = np.asarray(candidate[mask_key_name], dtype=np.float32).reshape(-1) > 0.5
                if values.shape[0] == overlay_mask.shape[0]:
                    overlay_mask |= values
        overlay_frames = int(np.count_nonzero(overlay_mask))
        if overlay_frames:
            dense[overlay_mask] = base_dense[overlay_mask]

    payload = dict(candidate)
    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["planned_hand_joints"] = dense[::substeps][: target_keys.shape[0]].astype(np.float32)
    payload["planned_hand_velocities"] = compute_hand_velocities(
        payload["planned_hand_joints"], control_timestep=0.05
    ).astype(np.float32)
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
        "source_npz": str(args.source_npz),
        "candidate_npz": str(args.candidate_npz),
        "output_dir": str(out),
        "candidate_rows": int(qpos_rows.shape[0]),
        "unique_masks": int(len({mask_key(row, float(args.threshold)) for row in target_keys})),
        "changed_control_frames": int(changed_control),
        "fallback_frames": int(fallback_frames),
        "event_f1": float(event_f1(score)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "played": int(score.get("played_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "max_static_fp": int(args.max_static_fp),
        "min_static_f1": float(args.min_static_f1),
        "rest_source": str(args.rest_source),
        "include_dense": bool(args.include_dense),
        "overlay_candidate_injections": bool(args.overlay_candidate_injections),
        "overlay_frames": int(overlay_frames),
        "selection_report": report,
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
