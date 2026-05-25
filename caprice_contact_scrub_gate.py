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
from impromptu.joint_space_trajectory import BAGATELLE_FINGER_JOINT_INDEX_ROWS  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


LEFT_FOREARM_INDICES = np.asarray([44, 45], dtype=np.int64)
RIGHT_FOREARM_INDICES = np.asarray([21, 22], dtype=np.int64)


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


def score_key(score: dict[str, Any]) -> tuple[float, int, int, float]:
    return (
        float(score["event_f1"]),
        -int(score["mispresses"]),
        int(score["matched"]),
        float(score["frame_f1"]),
    )


def simulate_score(
    payload: dict[str, np.ndarray],
    dense: np.ndarray,
    out: Path,
    env: str,
    threshold: float,
    dense_dt: float,
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
    score = score_rollout(
        target_keys=target,
        played_keys=played,
        dt=float(dense_dt),
        threshold=float(threshold),
        timing_tolerance_s=0.15,
    )
    return compact_score(score), score


def contact_counts(target: np.ndarray, activation: np.ndarray, threshold: float) -> tuple[int, int, int]:
    goal = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    played = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(goal, played).sum())
    fp = int(np.logical_and(~goal, played).sum())
    fn = int(np.logical_and(goal, ~played).sum())
    return tp, fp, fn


def contact_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * int(tp) + int(fp) + int(fn)
    return float(2 * int(tp) / denom) if denom else 1.0


def contact_rank(
    target: np.ndarray,
    activation: np.ndarray,
    threshold: float,
    *,
    fp_weight: float,
    fn_weight: float,
    false_key: int,
) -> tuple[float, int, int, int]:
    tp, fp, fn = contact_counts(target, activation, threshold)
    played = np.asarray(activation, dtype=np.float32)[:88] > float(threshold)
    false_penalty = 1 if 0 <= int(false_key) < 88 and bool(played[int(false_key)]) else 0
    value = contact_f1(tp, fp, fn) + 0.10 * tp - float(fp_weight) * fp - float(fn_weight) * fn - 1.5 * false_penalty
    return (float(value), int(tp), -int(fp), -int(fn))


def hand_forearm_indices(finger: int) -> np.ndarray:
    return LEFT_FOREARM_INDICES if int(finger) < 5 else RIGHT_FOREARM_INDICES


def selected_indices(finger: int, scope: str) -> np.ndarray:
    joints = list(BAGATELLE_FINGER_JOINT_INDEX_ROWS[int(finger)])
    if scope == "finger":
        return np.asarray(joints, dtype=np.int64)
    if scope == "hand":
        return np.unique(np.concatenate([np.asarray(joints, dtype=np.int64), hand_forearm_indices(int(finger))]))
    raise ValueError(f"unknown scope {scope!r}")


def nearest_fingers_for_key(
    kin: BagatelleKinematics,
    qpos: np.ndarray,
    key: int,
    top_fingers: int,
) -> list[dict[str, Any]]:
    fingertips = np.asarray(kin.fingertip_positions_for_qpos(qpos), dtype=np.float32)
    target = np.asarray(kin.key_contact_targets(np.asarray([int(key)], dtype=np.int32))[0], dtype=np.float32)
    distances = np.linalg.norm(fingertips[:, :2] - target[:2].reshape(1, 2), axis=1)
    order = np.argsort(distances, kind="stable")[: max(int(top_fingers), 1)]
    return [
        {
            "finger": int(finger),
            "xy_distance": float(distances[int(finger)]),
        }
        for finger in order.tolist()
    ]


def base_mode_qpos(
    dense: np.ndarray,
    *,
    frame: int,
    start: int,
    end: int,
    finger: int,
    mode: str,
    neutral: np.ndarray,
) -> np.ndarray:
    qpos = np.asarray(dense[int(frame)], dtype=np.float32).copy()
    joints = np.asarray(BAGATELLE_FINGER_JOINT_INDEX_ROWS[int(finger)], dtype=np.int64)
    if mode == "neutral":
        qpos[joints] = neutral[joints]
    elif mode == "zero":
        qpos[joints] = 0.0
    elif mode == "hold_before":
        state = dense[start - 1] if start > 0 else neutral
        qpos[joints] = state[joints]
    elif mode == "hold_after":
        state = dense[end] if end < dense.shape[0] else neutral
        qpos[joints] = state[joints]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return qpos.astype(np.float32)


def perturb_qpos(
    kin: BagatelleKinematics,
    base: np.ndarray,
    rng: np.random.Generator,
    *,
    finger: int,
    scope: str,
    finger_sigma: float,
    forearm_sigma: float,
) -> np.ndarray:
    out = np.asarray(base, dtype=np.float32).copy()
    joints = np.asarray(BAGATELLE_FINGER_JOINT_INDEX_ROWS[int(finger)], dtype=np.int64)
    out[joints] += rng.normal(0.0, float(finger_sigma), size=joints.size).astype(np.float32)
    if scope == "hand":
        forearm = hand_forearm_indices(int(finger))
        out[forearm] += rng.normal(0.0, float(forearm_sigma), size=forearm.size).astype(np.float32)
    return kin.clip_qpos(out)


def apply_candidate_slice(
    dense: np.ndarray,
    *,
    start: int,
    end: int,
    candidate_qpos: np.ndarray,
    finger: int,
    scope: str,
) -> np.ndarray:
    candidate = np.asarray(dense[start:end], dtype=np.float32).copy()
    indices = selected_indices(int(finger), str(scope))
    candidate[:, indices] = np.asarray(candidate_qpos, dtype=np.float32)[indices].reshape(1, -1)
    return candidate.astype(np.float32)


def build_static_candidates(
    *,
    kin: BagatelleKinematics,
    target_row: np.ndarray,
    dense: np.ndarray,
    frame: int,
    start: int,
    end: int,
    key: int,
    nearest: list[dict[str, Any]],
    neutral: np.ndarray,
    rng: np.random.Generator,
    threshold: float,
    modes: list[str],
    samples_per_finger: int,
    top_candidates: int,
    finger_sigma: float,
    forearm_sigma: float,
    fp_weight: float,
    fn_weight: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in nearest:
        finger = int(item["finger"])
        for mode in modes:
            qpos = kin.clip_qpos(
                base_mode_qpos(dense, frame=frame, start=start, end=end, finger=finger, mode=mode, neutral=neutral)
            )
            for scope in ("finger",):
                activation = kin.activation_for_qpos(qpos, settle_steps=1)[:88]
                rows.append(
                    {
                        "finger": finger,
                        "scope": scope,
                        "source": mode,
                        "qpos": qpos,
                        "activation": activation,
                        "rank": contact_rank(
                            target_row,
                            activation,
                            threshold,
                            fp_weight=fp_weight,
                            fn_weight=fn_weight,
                            false_key=key,
                        ),
                    }
                )
        base = np.asarray(dense[int(frame)], dtype=np.float32)
        for scope in ("finger", "hand"):
            for sample in range(max(int(samples_per_finger), 0)):
                qpos = perturb_qpos(
                    kin,
                    base,
                    rng,
                    finger=finger,
                    scope=scope,
                    finger_sigma=float(finger_sigma),
                    forearm_sigma=float(forearm_sigma),
                )
                activation = kin.activation_for_qpos(qpos, settle_steps=1)[:88]
                rows.append(
                    {
                        "finger": finger,
                        "scope": scope,
                        "source": f"random_{sample}",
                        "qpos": qpos,
                        "activation": activation,
                        "rank": contact_rank(
                            target_row,
                            activation,
                            threshold,
                            fp_weight=fp_weight,
                            fn_weight=fn_weight,
                            false_key=key,
                        ),
                    }
                )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for row in rows:
        indices = selected_indices(int(row["finger"]), str(row["scope"]))
        key_bytes = (
            str(row["finger"]).encode("utf-8")
            + b":"
            + str(row["scope"]).encode("utf-8")
            + b":"
            + np.round(np.asarray(row["qpos"], dtype=np.float32)[indices], 5).tobytes()
        )
        if key_bytes in seen:
            continue
        seen.add(key_bytes)
        tp, fp, fn = contact_counts(target_row, row["activation"], threshold)
        out.append(
            {
                "finger": int(row["finger"]),
                "scope": str(row["scope"]),
                "source": str(row["source"]),
                "qpos": np.asarray(row["qpos"], dtype=np.float32),
                "static_rank": float(row["rank"][0]),
                "static_tp": int(tp),
                "static_fp": int(fp),
                "static_fn": int(fn),
            }
        )
        if len(out) >= max(int(top_candidates), 1):
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Caprice: contact-aware local qpos scrub for false-key Impromptu events.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pre-dense-frames", type=int, default=1)
    parser.add_argument("--post-dense-frames", type=int, default=8)
    parser.add_argument("--top-fingers", type=int, default=2)
    parser.add_argument("--modes", default="neutral,hold_before,zero")
    parser.add_argument("--samples-per-finger", type=int, default=18)
    parser.add_argument("--top-candidates", type=int, default=4)
    parser.add_argument("--finger-sigma", type=float, default=0.035)
    parser.add_argument("--forearm-sigma", type=float, default=0.016)
    parser.add_argument("--static-fp-weight", type=float, default=0.30)
    parser.add_argument("--static-fn-weight", type=float, default=0.18)
    parser.add_argument("--max-events", type=int, default=45)
    parser.add_argument("--max-total-evals", type=int, default=160)
    parser.add_argument("--min-delta-event-f1", type=float, default=1e-6)
    parser.add_argument("--allow-equal-f1-mispress-drop", action="store_true")
    parser.add_argument("--seed", type=int, default=524)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.source_npz, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32).copy()
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    control_steps = int(target_keys.shape[0])
    substeps = max(int(dense.shape[0] // max(control_steps, 1)), 1)
    dense_dt = 0.05 / float(substeps)
    neutral = dense[0].astype(np.float32)
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    rng = np.random.default_rng(int(args.seed))

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
    scrubbed = np.zeros((dense.shape[0],), dtype=np.float32)

    config = BagatelleConfig(environment_name=str(args.environment_name), threshold=float(args.threshold), seed=0)
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        for event_index, event in enumerate(list(detail.get("mispress_events", []))[: max(int(args.max_events), 0)]):
            if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                break
            frame = int(event.get("frame", 0))
            key = int(event.get("key", -1))
            if key < 0 or key >= 88:
                continue
            start = max(frame - max(int(args.pre_dense_frames), 0), 0)
            end = min(frame + max(int(args.post_dense_frames), 0) + 1, dense.shape[0])
            if end <= start:
                continue
            previous = dense[start:end].copy()
            qpos = dense[min(max(frame, 0), dense.shape[0] - 1)]
            target_row = target_keys[min(max(frame // substeps, 0), target_keys.shape[0] - 1)]
            nearest = nearest_fingers_for_key(kin, qpos, key, int(args.top_fingers))
            static_candidates = build_static_candidates(
                kin=kin,
                target_row=target_row,
                dense=dense,
                frame=frame,
                start=start,
                end=end,
                key=key,
                nearest=nearest,
                neutral=neutral,
                rng=rng,
                threshold=float(args.threshold),
                modes=modes,
                samples_per_finger=int(args.samples_per_finger),
                top_candidates=int(args.top_candidates),
                finger_sigma=float(args.finger_sigma),
                forearm_sigma=float(args.forearm_sigma),
                fp_weight=float(args.static_fp_weight),
                fn_weight=float(args.static_fn_weight),
            )
            best_score: dict[str, Any] | None = None
            best_slice: np.ndarray | None = None
            best_candidate: dict[str, Any] | None = None
            trials: list[dict[str, Any]] = []
            for candidate_index, candidate in enumerate(static_candidates):
                if int(args.max_total_evals) > 0 and eval_count >= int(args.max_total_evals):
                    break
                dense[start:end] = apply_candidate_slice(
                    dense,
                    start=start,
                    end=end,
                    candidate_qpos=np.asarray(candidate["qpos"], dtype=np.float32),
                    finger=int(candidate["finger"]),
                    scope=str(candidate["scope"]),
                )
                eval_count += 1
                trial, _ = simulate_score(
                    payload,
                    dense,
                    out / f"eval_{eval_count:04d}",
                    str(args.environment_name),
                    float(args.threshold),
                    dense_dt,
                )
                trial_record = {
                    "candidate_index": int(candidate_index),
                    "finger": int(candidate["finger"]),
                    "scope": str(candidate["scope"]),
                    "source": str(candidate["source"]),
                    "static_rank": float(candidate["static_rank"]),
                    "static_tp": int(candidate["static_tp"]),
                    "static_fp": int(candidate["static_fp"]),
                    "static_fn": int(candidate["static_fn"]),
                    "trial": trial,
                }
                trials.append(trial_record)
                if best_score is None or score_key(trial) > score_key(best_score):
                    best_score = dict(trial)
                    best_slice = dense[start:end].copy()
                    best_candidate = trial_record
                dense[start:end] = previous

            delta = float(best_score["event_f1"] - current["event_f1"]) if best_score else 0.0
            accept = bool(best_score) and delta >= float(args.min_delta_event_f1)
            if (
                bool(best_score)
                and bool(args.allow_equal_f1_mispress_drop)
                and delta >= -1e-12
                and int(best_score["mispresses"]) < int(current["mispresses"])
            ):
                accept = True
            record = {
                "event_index": int(event_index),
                "event": {"frame": int(frame), "key": int(key), "time_s": float(event.get("time_s", 0.0))},
                "dense_start": int(start),
                "dense_end": int(end),
                "nearest": nearest,
                "previous": current,
                "best_trial": best_score,
                "best_candidate": best_candidate,
                "delta_event_f1": float(delta),
                "static_candidates": [
                    {
                        "finger": int(row["finger"]),
                        "scope": str(row["scope"]),
                        "source": str(row["source"]),
                        "static_rank": float(row["static_rank"]),
                        "static_tp": int(row["static_tp"]),
                        "static_fp": int(row["static_fp"]),
                        "static_fn": int(row["static_fn"]),
                    }
                    for row in static_candidates
                ],
                "trials": trials,
            }
            if accept and best_slice is not None and best_score is not None:
                dense[start:end] = best_slice
                scrubbed[start:end] = 1.0
                current = best_score
                accepted.append(record)
            else:
                dense[start:end] = previous
                rejected.append(record)

    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["planned_hand_joints"] = dense[::substeps][:control_steps].astype(np.float32)
    payload["planned_hand_velocities"] = compute_hand_velocities(payload["planned_hand_joints"], control_timestep=0.05)
    payload["caprice_scrubbed_dense_frames"] = scrubbed
    atomic_save_npz(out / "trajectory.npz", **payload)
    final_score, _ = simulate_score(payload, dense, out / "rp1m_sim", str(args.environment_name), float(args.threshold), dense_dt)
    result = {
        "source_npz": str(args.source_npz),
        "output_dir": str(out),
        "pre_dense_frames": int(args.pre_dense_frames),
        "post_dense_frames": int(args.post_dense_frames),
        "top_fingers": int(args.top_fingers),
        "modes": modes,
        "samples_per_finger": int(args.samples_per_finger),
        "top_candidates": int(args.top_candidates),
        "finger_sigma": float(args.finger_sigma),
        "forearm_sigma": float(args.forearm_sigma),
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
