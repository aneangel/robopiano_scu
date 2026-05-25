#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
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
    REPO_ROOT / "Rhapsody" / "src",
):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from bagatelle.assignment import FingerAssignmentResult  # noqa: E402
from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402
from impromptu.joint_space_trajectory import BAGATELLE_FINGER_JOINT_INDEX_ROWS  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402
from rhapsody.solver import RhapsodyIKSolver  # noqa: E402


RIGHT_HAND = np.arange(0, 23, dtype=np.int64)
LEFT_HAND = np.arange(23, 46, dtype=np.int64)
FULL_HAND = np.arange(46, dtype=np.int64)


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def compact(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_f1": float(event_f1(score)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "played": int(score.get("played_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
    }


def score_key(score: dict[str, Any]) -> tuple[float, int, int, float]:
    return (float(score["event_f1"]), int(score["matched"]), -int(score["mispresses"]), float(score["frame_f1"]))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def simulate(
    payload: dict[str, np.ndarray],
    dense: np.ndarray,
    out: Path,
    env: str,
    threshold: float,
    dense_dt: float,
    current: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    substeps = max(int(dense.shape[0] // max(target_keys.shape[0], 1)), 1)
    goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(env),
        demo_id=0,
        actions=np.zeros((dense.shape[0], 39), dtype=np.float32),
        goals=goals,
        hand_joints=np.asarray(dense, dtype=np.float32),
        environment_name=str(env),
    )
    try:
        sim = simulate_rp1m_rollout(
            traj,
            RolloutConfig(
                mode="hand_state",
                dataset_timestep=float(dense_dt),
                simulation_timestep=float(dense_dt),
                hand_anchor_y_offset=None,
                hand_state_action_source="zero",
                restore_initial_hand=True,
                set_hand_qvel=False,
                threshold=float(threshold),
                render_mp4=False,
                render_audio=False,
            ),
            out,
        )
        with np.load(sim["rollout_npz"], allow_pickle=False) as data:
            played = np.asarray(data["source_played_piano"], dtype=np.float32)[:, :88]
            target = np.asarray(data["goals"], dtype=np.float32)[:, :88]
        detail = score_rollout(target_keys=target, played_keys=played, dt=float(dense_dt), threshold=float(threshold), timing_tolerance_s=0.15)
        return compact(detail), detail
    except Exception as exc:
        fallback = {
            "event_f1": -1.0,
            "frame_f1": -1.0,
            "matched": int((current or {}).get("matched", 0)),
            "target": int((current or {}).get("target", 0)),
            "played": 10**9,
            "mispresses": 10**9,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return fallback, {"error": fallback["error"], "missed_events": [], "mispress_events": []}


def assignment_for_chord(kin: BagatelleKinematics, keys: np.ndarray, fingers: np.ndarray) -> FingerAssignmentResult:
    active_keys = np.asarray(keys, dtype=np.int32).reshape(-1)
    finger_indices = np.asarray(fingers, dtype=np.int32).reshape(-1)
    targets = kin.key_press_targets(active_keys)
    positions = np.arange(active_keys.size, dtype=np.int32)
    return FingerAssignmentResult(
        active_keys=active_keys,
        assigned_finger_indices=finger_indices,
        assigned_keys=active_keys.copy(),
        assigned_key_positions=positions,
        target_positions=targets.astype(np.float32),
        unassigned_keys=np.zeros((0,), dtype=np.int32),
        cost_matrix=np.zeros((10, active_keys.size), dtype=np.float32),
        total_cost=0.0,
        mean_cost=0.0,
        strategy="polka_contact_mpc_chord",
    )


def chord_finger_candidates(keys: np.ndarray, split_key: int, max_patterns: int) -> list[np.ndarray]:
    values = np.asarray(keys, dtype=np.int32).reshape(-1)
    if values.size == 1:
        return [np.asarray([finger], dtype=np.int32) for finger in range(10)][: max(int(max_patterns), 1)]
    low_positions = np.flatnonzero(values < int(split_key))
    high_positions = np.flatnonzero(values >= int(split_key))
    patterns: list[np.ndarray] = []

    def hand_patterns(count: int, hand: str) -> list[tuple[int, ...]]:
        if count <= 0:
            return [()]
        if hand == "left":
            base = [4, 3, 2, 1, 0]
        else:
            base = [5, 6, 7, 8, 9]
        return list(itertools.permutations(base, count))[: max(int(max_patterns), 1)]

    for left_fingers in hand_patterns(low_positions.size, "left"):
        for right_fingers in hand_patterns(high_positions.size, "right"):
            row = np.full(values.shape, -1, dtype=np.int32)
            for pos, finger in zip(low_positions.tolist(), left_fingers):
                row[pos] = int(finger)
            for pos, finger in zip(high_positions.tolist(), right_fingers):
                row[pos] = int(finger)
            if np.all(row >= 0):
                patterns.append(row.astype(np.int32))
            if len(patterns) >= int(max_patterns):
                return patterns
    return patterns


def effective_split_key(keys: np.ndarray, default_split_key: int, auto_split_span: int) -> int:
    values = np.sort(np.asarray(keys, dtype=np.int32).reshape(-1))
    if values.size < 2 or int(auto_split_span) <= 0:
        return int(default_split_key)
    if int(values[-1] - values[0]) < int(auto_split_span):
        return int(default_split_key)
    gaps = np.diff(values)
    if gaps.size == 0:
        return int(default_split_key)
    gap_index = int(np.argmax(gaps))
    if int(gaps[gap_index]) <= 0:
        return int(default_split_key)
    return int((int(values[gap_index]) + int(values[gap_index + 1]) + 1) // 2)


def scope_indices(keys: np.ndarray, fingers: np.ndarray, scope: str, split_key: int) -> np.ndarray:
    if scope == "full":
        return FULL_HAND
    if scope == "hand":
        parts = []
        if np.any(np.asarray(keys, dtype=np.int32) < int(split_key)):
            parts.append(LEFT_HAND)
        if np.any(np.asarray(keys, dtype=np.int32) >= int(split_key)):
            parts.append(RIGHT_HAND)
        return np.unique(np.concatenate(parts)).astype(np.int64) if parts else FULL_HAND
    if scope == "finger":
        parts = [np.asarray(BAGATELLE_FINGER_JOINT_INDEX_ROWS[int(finger)], dtype=np.int64) for finger in np.asarray(fingers, dtype=np.int32)]
        return np.unique(np.concatenate(parts)).astype(np.int64) if parts else FULL_HAND
    raise ValueError(f"unknown scope {scope!r}")


def rhapsody_pose(
    solver: RhapsodyIKSolver,
    *,
    kin: BagatelleKinematics,
    keys: np.ndarray,
    fingers: np.ndarray,
    qpos: np.ndarray,
    refinement_steps: int,
) -> dict[str, Any]:
    active_keys = np.asarray(keys, dtype=np.int32).reshape(-1)
    finger_ids = np.asarray(fingers, dtype=np.int32).reshape(-1)
    targets = kin.key_press_targets(active_keys).astype(np.float32)
    target_tips = np.full((10, 3), np.nan, dtype=np.float32)
    active_mask = np.zeros((10,), dtype=np.float32)
    for key_position, finger in enumerate(finger_ids.tolist()):
        if 0 <= int(finger) < 10:
            target_tips[int(finger)] = targets[int(key_position)]
            active_mask[int(finger)] = 1.0
    solution = solver.solve(
        target_tips,
        active_mask=active_mask,
        previous_qpos=np.asarray(qpos, dtype=np.float32),
        refinement_steps=int(refinement_steps),
    )
    return {
        "pose": np.asarray(solution.qpos, dtype=np.float32),
        "source": "rhapsody",
        "rhapsody_mean_error_m": float(solution.mean_error_m),
        "rhapsody_max_error_m": float(solution.max_error_m),
        "rhapsody_refinement_steps": int(solution.refinement_steps),
        "ik_success": bool(np.isfinite(solution.qpos).all()),
        "ik_max_residual": float(solution.max_error_m),
        "ik_residual_norm": float(solution.mean_error_m),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Polka: chord-level contact-MPC synthesis for clustered missed events.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-key", type=int, default=48)
    parser.add_argument("--auto-split-span", type=int, default=0)
    parser.add_argument("--pre-dense-frames", type=int, default=1)
    parser.add_argument("--post-dense-frames", type=int, default=7)
    parser.add_argument("--missed-only", action="store_true")
    parser.add_argument("--key-source", choices=("active", "missed", "press"), default="active")
    parser.add_argument("--scopes", default="full")
    parser.add_argument("--rhapsody-checkpoint", default="")
    parser.add_argument("--rhapsody-refinement-steps", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=40)
    parser.add_argument("--max-patterns", type=int, default=24)
    parser.add_argument("--max-total-evals", type=int, default=120)
    parser.add_argument("--min-delta-event-f1", type=float, default=1e-6)
    parser.add_argument("--keep-if-matched-increases", action="store_true")
    parser.add_argument("--matched-gain-max-f1-drop", type=float, default=0.002)
    parser.add_argument("--allow-equal-f1-mispress-drop", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = load_npz(Path(args.source_npz))
    dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32).copy()
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control_steps = int(target_keys.shape[0])
    substeps = max(int(dense.shape[0] // max(control_steps, 1)), 1)
    dense_dt = 0.05 / float(substeps)
    target_dense = np.repeat(target_keys > float(args.threshold), substeps, axis=0)[: dense.shape[0], :88]
    press_dense = target_dense & ~np.vstack([np.zeros((1, 88), dtype=bool), target_dense[:-1]])
    scopes = [item.strip() for item in str(args.scopes).split(",") if item.strip()]

    current, detail = simulate(payload, dense, out / "eval_000_baseline", str(args.environment_name), float(args.threshold), dense_dt)
    initial = dict(current)
    missed_frames = []
    missed_by_frame: dict[int, list[int]] = {}
    seen = set()
    for event in detail.get("missed_events", []):
        frame = int(event.get("frame", -1))
        key = int(event.get("key", -1))
        if 0 <= frame < dense.shape[0] and frame not in seen:
            seen.add(frame)
            missed_frames.append(frame)
        if 0 <= frame < dense.shape[0] and 0 <= key < 88:
            missed_by_frame.setdefault(frame, []).append(key)
        if len(missed_frames) >= max(int(args.max_events), 1):
            break

    bag_cfg = BagatelleConfig(
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        seed=0,
        control_timestep=0.05,
        ik_static_contact_validation=True,
        ik_static_contact_wrong_key_weight=0.5,
        ik_static_contact_missed_key_weight=4.0,
        ik_static_contact_settle_steps=1,
        ik_multistart_on_failure=True,
        ik_multistart_seed_count=8,
        ik_multistart_forearm_tx_grid=7,
        ik_unassigned_fingertip_strategy="avoid_mispresses",
        ik_unassigned_fingertip_avoidance_weight=0.7,
        ik_unassigned_fingertip_avoidance_radius=0.035,
        key_press_depth=0.006,
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    eval_count = 0
    rhapsody_solver = (
        RhapsodyIKSolver.from_checkpoint(Path(args.rhapsody_checkpoint), device="cpu")
        if str(args.rhapsody_checkpoint).strip()
        else None
    )
    with BagatelleKinematics(config=bag_cfg, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        neutral = np.asarray(kin.neutral_qpos, dtype=np.float32)
        for event_index, frame in enumerate(missed_frames):
            if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                break
            key_source = "missed" if bool(args.missed_only) else str(args.key_source)
            if key_source == "missed":
                keys = np.asarray(sorted(set(missed_by_frame.get(int(frame), []))), dtype=np.int32)
            elif key_source == "press":
                keys = np.flatnonzero(press_dense[int(frame)]).astype(np.int32)
            else:
                keys = np.flatnonzero(target_dense[int(frame)]).astype(np.int32)
            if keys.size == 0:
                continue
            keys.sort()
            start = max(int(frame) - max(int(args.pre_dense_frames), 0), 0)
            end = min(int(frame) + max(int(args.post_dense_frames), 0) + 1, dense.shape[0])
            previous_slice = dense[start:end].copy()
            qpos = dense[min(max(int(frame), 0), dense.shape[0] - 1)].copy()
            event_split_key = effective_split_key(keys, int(args.split_key), int(args.auto_split_span))
            patterns = chord_finger_candidates(keys, int(event_split_key), int(args.max_patterns))
            best_score: dict[str, Any] | None = None
            best_slice: np.ndarray | None = None
            best_candidate: dict[str, Any] | None = None
            trials: list[dict[str, Any]] = []
            for pattern_index, fingers in enumerate(patterns):
                if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                    break
                assignment = assignment_for_chord(kin, keys, fingers)
                result = kin.solve_press_pose(assignment, previous_qpos=qpos, neutral_qpos=neutral, config=bag_cfg)
                pose_rows: list[dict[str, Any]] = [
                    {
                        "pose": np.asarray(result.pose, dtype=np.float32),
                        "source": "bagatelle",
                        "ik_success": bool(result.success),
                        "ik_max_residual": float(result.max_residual),
                        "ik_residual_norm": float(result.residual_norm),
                    }
                ]
                if rhapsody_solver is not None:
                    try:
                        pose_rows.append(
                            rhapsody_pose(
                                rhapsody_solver,
                                kin=kin,
                                keys=keys,
                                fingers=fingers,
                                qpos=qpos,
                                refinement_steps=int(args.rhapsody_refinement_steps),
                            )
                        )
                    except Exception as exc:
                        pose_rows.append(
                            {
                                "pose": np.asarray(result.pose, dtype=np.float32),
                                "source": "rhapsody_error",
                                "ik_success": False,
                                "ik_max_residual": 1.0e9,
                                "ik_residual_norm": 1.0e9,
                                "rhapsody_error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                for pose_row in pose_rows:
                    pose = np.asarray(pose_row["pose"], dtype=np.float32)
                    activation = kin.activation_for_qpos(pose, settle_steps=1)[:88]
                    played_static = np.flatnonzero(activation > float(args.threshold)).astype(int)
                    static_hits = sorted(set(played_static.tolist()) & set(keys.astype(int).tolist()))
                    wrong_static = sorted(set(played_static.tolist()) - set(keys.astype(int).tolist()))
                    for scope in scopes:
                        if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                            break
                        indices = scope_indices(keys, fingers, str(scope), int(event_split_key))
                        dense[start:end] = previous_slice
                        dense[start:end, indices] = pose[indices].reshape(1, -1)
                        eval_count += 1
                        trial, _ = simulate(
                            payload,
                            dense,
                            out / f"eval_{eval_count:04d}",
                            str(args.environment_name),
                            float(args.threshold),
                            dense_dt,
                            current=current,
                        )
                        candidate = {
                            "pattern_index": int(pattern_index),
                            "keys": keys.astype(int).tolist(),
                            "fingers": np.asarray(fingers, dtype=int).tolist(),
                            "scope": str(scope),
                            "source": str(pose_row.get("source", "unknown")),
                            "split_key": int(event_split_key),
                            "static_hits": static_hits,
                            "static_wrong": wrong_static,
                            "ik_success": bool(pose_row.get("ik_success", False)),
                            "ik_max_residual": float(pose_row.get("ik_max_residual", 1.0e9)),
                            "ik_residual_norm": float(pose_row.get("ik_residual_norm", 1.0e9)),
                            "trial": trial,
                        }
                        for key_name in ("rhapsody_mean_error_m", "rhapsody_max_error_m", "rhapsody_refinement_steps", "rhapsody_error"):
                            if key_name in pose_row:
                                candidate[key_name] = pose_row[key_name]
                        trials.append(candidate)
                        if best_score is None or score_key(trial) > score_key(best_score):
                            best_score = dict(trial)
                            best_slice = dense[start:end].copy()
                            best_candidate = candidate
                        dense[start:end] = previous_slice
                    if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                        break
                if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                    break
            delta = float(best_score["event_f1"] - current["event_f1"]) if best_score else 0.0
            accept = bool(best_score) and delta >= float(args.min_delta_event_f1)
            if (
                bool(best_score)
                and bool(args.keep_if_matched_increases)
                and int(best_score["matched"]) > int(current["matched"])
                and delta >= -abs(float(args.matched_gain_max_f1_drop))
            ):
                accept = True
            if (
                bool(best_score)
                and bool(args.allow_equal_f1_mispress_drop)
                and delta >= -1e-12
                and int(best_score["mispresses"]) < int(current["mispresses"])
            ):
                accept = True
            record = {
                "event_index": int(event_index),
                "frame": int(frame),
                "dense_start": int(start),
                "dense_end": int(end),
                "keys": keys.astype(int).tolist(),
                "split_key": int(event_split_key),
                "previous": dict(current),
                "best_candidate": best_candidate,
                "best_trial": best_score,
                "delta_event_f1": float(delta),
                "trials": trials,
            }
            if accept and best_slice is not None and best_score is not None:
                dense[start:end] = best_slice
                current = best_score
                accepted.append(record)
            else:
                dense[start:end] = previous_slice
                rejected.append(record)

    output = dict(payload)
    output["planned_hand_joints_dense"] = dense.astype(np.float32)
    output["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    output["planned_hand_joints"] = dense[::substeps][:control_steps].astype(np.float32)
    output["planned_hand_velocities"] = compute_hand_velocities(output["planned_hand_joints"], control_timestep=0.05)
    atomic_save_npz(out / "trajectory.npz", **output)
    final_score, _ = simulate(payload, dense, out / "rp1m_sim", str(args.environment_name), float(args.threshold), dense_dt, current=current)
    result = {
        "source_npz": str(args.source_npz),
        "output_dir": str(out),
        "initial": initial,
        "greedy_score": current,
        "final_score": final_score,
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "eval_count": int(eval_count),
        "accepted": accepted[:300],
        "rejected": rejected[:300],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
