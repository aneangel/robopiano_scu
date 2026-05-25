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


def event_f1_from_score(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def compact_score(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_f1": float(event_f1_from_score(score)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "played": int(score.get("played_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
    }


def bad_score(current: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "event_f1": -1.0,
        "frame_f1": -1.0,
        "matched": int(current.get("matched", 0)),
        "target": int(current.get("target", 0)),
        "played": 10**9,
        "mispresses": 10**9,
        "error": f"{type(error).__name__}: {error}",
    }


def score_key(score: dict[str, Any]) -> tuple[float, int, int, float]:
    return (
        float(score["event_f1"]),
        int(score["matched"]),
        -int(score["mispresses"]),
        float(score["frame_f1"]),
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def simulate_score(
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
        detail = score_rollout(
            target_keys=target,
            played_keys=played,
            dt=float(dense_dt),
            threshold=float(threshold),
            timing_tolerance_s=0.15,
        )
        return compact_score(detail), detail
    except Exception as exc:
        fallback = bad_score(current or {}, exc)
        return fallback, {"error": fallback["error"], "missed_events": [], "mispress_events": []}


def fingers_for_key(key: int, split_key: int, mode: str) -> list[int]:
    if mode == "all":
        return list(range(10))
    if int(key) < int(split_key):
        return [1, 2, 3, 4, 0]
    return [6, 7, 8, 9, 5]


def indices_for_scope(finger: int, scope: str, key: int, split_key: int) -> np.ndarray:
    if scope == "finger":
        return np.asarray(BAGATELLE_FINGER_JOINT_INDEX_ROWS[int(finger)], dtype=np.int64)
    if scope == "hand":
        return LEFT_HAND if int(key) < int(split_key) else RIGHT_HAND
    if scope == "full":
        return FULL_HAND
    raise ValueError(f"unknown scope {scope!r}")


def assignment_for_key(kin: BagatelleKinematics, key: int, finger: int) -> FingerAssignmentResult:
    active_keys = np.asarray([int(key)], dtype=np.int32)
    targets = kin.key_press_targets(active_keys)
    return FingerAssignmentResult(
        active_keys=active_keys,
        assigned_finger_indices=np.asarray([int(finger)], dtype=np.int32),
        assigned_keys=active_keys.copy(),
        assigned_key_positions=np.asarray([0], dtype=np.int32),
        target_positions=targets.astype(np.float32),
        unassigned_keys=np.zeros((0,), dtype=np.int32),
        cost_matrix=np.zeros((10, 1), dtype=np.float32),
        total_cost=0.0,
        mean_cost=0.0,
        strategy="waltz_contact_mpc_single_key",
    )


def solve_contact_candidates(
    *,
    kin: BagatelleKinematics,
    key: int,
    qpos: np.ndarray,
    neutral: np.ndarray,
    fingers: list[int],
    scopes: list[str],
    config: BagatelleConfig,
    threshold: float,
    rhapsody_solver: RhapsodyIKSolver | None = None,
    rhapsody_refinement_steps: int = 16,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for finger in fingers:
        assignment = assignment_for_key(kin, int(key), int(finger))
        result = kin.solve_press_pose(assignment, previous_qpos=qpos, neutral_qpos=neutral, config=config)
        pose_rows: list[dict[str, Any]] = [
            {
                "source": "bagatelle",
                "pose": np.asarray(result.pose, dtype=np.float32).copy(),
                "ik_success": bool(result.success),
                "ik_max_residual": float(result.max_residual),
                "ik_residual_norm": float(result.residual_norm),
            }
        ]
        if rhapsody_solver is not None:
            target_tips = np.full((10, 3), np.nan, dtype=np.float32)
            active_mask = np.zeros((10,), dtype=np.float32)
            target_tips[int(finger)] = kin.key_press_targets(np.asarray([int(key)], dtype=np.int32))[0]
            active_mask[int(finger)] = 1.0
            try:
                solution = rhapsody_solver.solve(
                    target_tips,
                    active_mask=active_mask,
                    previous_qpos=np.asarray(qpos, dtype=np.float32),
                    refinement_steps=int(rhapsody_refinement_steps),
                )
                pose_rows.append(
                    {
                        "source": "rhapsody",
                        "pose": np.asarray(solution.qpos, dtype=np.float32).copy(),
                        "ik_success": bool(np.isfinite(solution.qpos).all()),
                        "ik_max_residual": float(solution.max_error_m),
                        "ik_residual_norm": float(solution.mean_error_m),
                        "rhapsody_mean_error_m": float(solution.mean_error_m),
                        "rhapsody_max_error_m": float(solution.max_error_m),
                        "rhapsody_refinement_steps": int(solution.refinement_steps),
                    }
                )
            except Exception as exc:
                pose_rows.append(
                    {
                        "source": "rhapsody_error",
                        "pose": np.asarray(result.pose, dtype=np.float32).copy(),
                        "ik_success": False,
                        "ik_max_residual": 1.0e9,
                        "ik_residual_norm": 1.0e9,
                        "rhapsody_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        for pose_row in pose_rows:
            pose = np.asarray(pose_row["pose"], dtype=np.float32)
            activation = kin.activation_for_qpos(pose, settle_steps=max(int(config.ik_static_contact_settle_steps), 1))[:88]
            played = np.flatnonzero(activation > float(threshold)).astype(int)
            hit = bool(0 <= int(key) < 88 and activation[int(key)] > float(threshold))
            wrong = int(np.count_nonzero(np.asarray([k for k in played.tolist() if int(k) != int(key)], dtype=np.int32)))
            rank = (
                1 if hit else 0,
                -wrong,
                -int(played.size),
                -float(pose_row["ik_max_residual"]),
                -float(pose_row["ik_residual_norm"]),
            )
            for scope in scopes:
                row = {
                    "finger": int(finger),
                    "scope": str(scope),
                    "source": str(pose_row["source"]),
                    "pose": pose.copy(),
                    "rank": rank,
                    "static_hit": bool(hit),
                    "static_wrong": int(wrong),
                    "static_played": played.astype(int).tolist(),
                    "ik_success": bool(pose_row["ik_success"]),
                    "ik_max_residual": float(pose_row["ik_max_residual"]),
                    "ik_residual_norm": float(pose_row["ik_residual_norm"]),
                }
                for key_name in ("rhapsody_mean_error_m", "rhapsody_max_error_m", "rhapsody_refinement_steps", "rhapsody_error"):
                    if key_name in pose_row:
                        row[key_name] = pose_row[key_name]
                rows.append(row)
    rows.sort(key=lambda row: row["rank"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Waltz: contact-MPC press-pose synthesis for missed Impromptu events.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-key", type=int, default=48)
    parser.add_argument("--finger-mode", choices=("hand", "all"), default="hand")
    parser.add_argument("--scopes", default="hand,full")
    parser.add_argument("--pre-dense-frames", type=int, default=1)
    parser.add_argument("--post-dense-frames", type=int, default=5)
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--max-candidates-per-event", type=int, default=10)
    parser.add_argument("--max-total-evals", type=int, default=180)
    parser.add_argument("--min-delta-event-f1", type=float, default=1e-6)
    parser.add_argument("--keep-if-matched-increases", action="store_true")
    parser.add_argument("--matched-gain-max-f1-drop", type=float, default=0.002)
    parser.add_argument("--allow-equal-f1-mispress-drop", action="store_true")
    parser.add_argument("--receding", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--rhapsody-checkpoint", default="")
    parser.add_argument("--rhapsody-refinement-steps", type=int, default=16)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = load_npz(Path(args.source_npz))
    dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32).copy()
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control_steps = int(target_keys.shape[0])
    substeps = max(int(dense.shape[0] // max(control_steps, 1)), 1)
    dense_dt = 0.05 / float(substeps)
    scopes = [item.strip() for item in str(args.scopes).split(",") if item.strip()]

    current, detail = simulate_score(
        payload,
        dense,
        out / "eval_000_baseline",
        str(args.environment_name),
        float(args.threshold),
        dense_dt,
    )
    initial = dict(current)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    eval_count = 0
    rhapsody_solver = (
        RhapsodyIKSolver.from_checkpoint(Path(args.rhapsody_checkpoint), device="cpu")
        if str(args.rhapsody_checkpoint).strip()
        else None
    )

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
        ik_multistart_seed_count=6,
        ik_multistart_forearm_tx_grid=7,
        ik_unassigned_fingertip_strategy="avoid_mispresses",
        ik_unassigned_fingertip_avoidance_weight=0.7,
        ik_unassigned_fingertip_avoidance_radius=0.035,
        key_press_depth=0.006,
    )
    with BagatelleKinematics(config=bag_cfg, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        neutral = np.asarray(kin.neutral_qpos, dtype=np.float32)
        rejected_events: set[tuple[int, int]] = set()
        rounds = max(int(args.max_rounds), 1) if bool(args.receding) else 1
        for round_index in range(rounds):
            accepted_this_round = False
            events = list(detail.get("missed_events", []))[: max(int(args.max_events), 0)]
            for event_index, event in enumerate(events):
                if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                    break
                frame = int(event.get("frame", -1))
                key = int(event.get("key", -1))
                if bool(args.receding) and (frame, key) in rejected_events:
                    continue
                if key < 0 or key >= 88 or frame < 0 or frame >= dense.shape[0]:
                    continue
                start = max(frame - max(int(args.pre_dense_frames), 0), 0)
                end = min(frame + max(int(args.post_dense_frames), 0) + 1, dense.shape[0])
                if end <= start:
                    continue
                previous_slice = dense[start:end].copy()
                qpos = dense[min(max(frame, 0), dense.shape[0] - 1)].copy()
                fingers = fingers_for_key(key, int(args.split_key), str(args.finger_mode))
                contact_candidates = solve_contact_candidates(
                    kin=kin,
                    key=key,
                    qpos=qpos,
                    neutral=neutral,
                    fingers=fingers,
                    scopes=scopes,
                    config=bag_cfg,
                    threshold=float(args.threshold),
                    rhapsody_solver=rhapsody_solver,
                    rhapsody_refinement_steps=int(args.rhapsody_refinement_steps),
                )
                best_score: dict[str, Any] | None = None
                best_detail: dict[str, Any] | None = None
                best_slice: np.ndarray | None = None
                best_candidate: dict[str, Any] | None = None
                trials: list[dict[str, Any]] = []
                for candidate in contact_candidates[: max(int(args.max_candidates_per_event), 1)]:
                    if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                        break
                    indices = indices_for_scope(int(candidate["finger"]), str(candidate["scope"]), key, int(args.split_key))
                    dense[start:end] = previous_slice
                    dense[start:end, indices] = np.asarray(candidate["pose"], dtype=np.float32)[indices].reshape(1, -1)
                    eval_count += 1
                    trial, trial_detail = simulate_score(
                        payload,
                        dense,
                        out / f"eval_{eval_count:04d}",
                        str(args.environment_name),
                        float(args.threshold),
                        dense_dt,
                        current=current,
                    )
                    row = {k: v for k, v in candidate.items() if k != "pose"}
                    row["trial"] = trial
                    trials.append(row)
                    if best_score is None or score_key(trial) > score_key(best_score):
                        best_score = dict(trial)
                        best_detail = trial_detail
                        best_slice = dense[start:end].copy()
                        best_candidate = row
                dense[start:end] = previous_slice
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
                    "round_index": int(round_index),
                    "event_index": int(event_index),
                    "event": {"frame": int(frame), "key": int(key)},
                    "dense_start": int(start),
                    "dense_end": int(end),
                    "previous": dict(current),
                    "best_candidate": best_candidate,
                    "best_trial": best_score,
                    "delta_event_f1": float(delta),
                    "trials": trials,
                }
                if accept and best_slice is not None and best_score is not None:
                    dense[start:end] = best_slice
                    current = best_score
                    if best_detail is not None:
                        detail = best_detail
                    accepted.append(record)
                    accepted_this_round = True
                    if bool(args.receding):
                        break
                else:
                    rejected.append(record)
                    if bool(args.receding):
                        rejected_events.add((frame, key))
            if not bool(args.receding) or not accepted_this_round:
                break

    output_payload = dict(payload)
    output_payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    output_payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    output_payload["planned_hand_joints"] = dense[::substeps][:control_steps].astype(np.float32)
    output_payload["planned_hand_velocities"] = compute_hand_velocities(output_payload["planned_hand_joints"], control_timestep=0.05)
    atomic_save_npz(out / "trajectory.npz", **output_payload)
    final_score, _ = simulate_score(
        payload,
        dense,
        out / "rp1m_sim",
        str(args.environment_name),
        float(args.threshold),
        dense_dt,
        current=current,
    )
    result = {
        "source_npz": str(args.source_npz),
        "output_dir": str(out),
        "initial": initial,
        "greedy_score": current,
        "final_score": final_score,
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "eval_count": int(eval_count),
        "accepted": accepted[:400],
        "rejected": rejected[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
