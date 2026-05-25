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


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def perturb(
    base: np.ndarray,
    rng: np.random.Generator,
    *,
    finger_sigma: float,
    forearm_sigma: float,
    include_forearm: bool,
) -> np.ndarray:
    cand = np.asarray(base, dtype=np.float32).copy()
    cand[ALL_FINGER_JOINT_INDICES] += rng.normal(
        0.0, float(finger_sigma), size=ALL_FINGER_JOINT_INDICES.size
    ).astype(np.float32)
    if include_forearm:
        cand[FOREARM_INDICES] += rng.normal(0.0, float(forearm_sigma), size=FOREARM_INDICES.size).astype(np.float32)
    return cand


def unique_candidates(
    rows: list[tuple[np.ndarray, np.ndarray, str]],
    *,
    max_candidates: int,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    seen: set[bytes] = set()
    for qpos, activation, source in rows:
        key = np.round(np.asarray(qpos, dtype=np.float32), 5).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((np.asarray(qpos, dtype=np.float32), np.asarray(activation, dtype=np.float32), source))
        if len(out) >= max(int(max_candidates), 1):
            break
    return out


def candidate_rank(target: np.ndarray, activation: np.ndarray, threshold: float, *, fp_weight: float) -> tuple[float, int, int, int]:
    tp, fp, fn = contact_counts(target, activation, threshold)
    f1 = contact_f1(target, activation, threshold)
    return (float(f1) + 0.08 * tp - float(fp_weight) * fp - 0.02 * fn, tp, -fp, -fn)


def build_candidate_pools(
    *,
    kin: BagatelleKinematics,
    target_keys: np.ndarray,
    control_qpos: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    samples_per_mask: int,
    iterations: int,
    elite_count: int,
    pool_size: int,
    top_seeds: int,
    finger_sigma: float,
    forearm_sigma: float,
    sigma_decay: float,
    fp_rank_weight: float,
    include_forearm: bool,
) -> tuple[dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray, str]]], list[dict[str, Any]]]:
    masks: dict[tuple[int, ...], list[int]] = {}
    for frame, row in enumerate(target_keys):
        masks.setdefault(mask_key(row, threshold), []).append(int(frame))

    pools: dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray, str]]] = {}
    reports: list[dict[str, Any]] = []
    for key, frames in sorted(masks.items(), key=lambda item: (len(item[0]), item[0])):
        target = target_keys[frames[0]]
        seeds: list[tuple[np.ndarray, np.ndarray, str]] = []
        for frame in frames:
            qpos = control_qpos[frame].copy()
            activation = kin.activation_for_qpos(qpos, settle_steps=1)[:88]
            seeds.append((qpos, activation, f"frame_{frame}"))
        seeds.sort(key=lambda item: candidate_rank(target, item[1], threshold, fp_weight=fp_rank_weight), reverse=True)
        candidates = seeds[: max(int(top_seeds), 1)]
        centers = [row[0] for row in candidates]
        fs = float(finger_sigma)
        forearm_s = float(forearm_sigma)
        for iteration in range(max(int(iterations), 0)):
            sampled: list[tuple[np.ndarray, np.ndarray, str]] = []
            for sample in range(max(int(samples_per_mask), 0)):
                center = centers[int(sample % len(centers))]
                qpos = kin.clip_qpos(
                    perturb(
                        center,
                        rng,
                        finger_sigma=fs,
                        forearm_sigma=forearm_s,
                        include_forearm=include_forearm,
                    )
                )
                activation = kin.activation_for_qpos(qpos, settle_steps=1)[:88]
                sampled.append((qpos, activation, f"iter{iteration}_sample{sample}"))
            candidates.extend(sampled)
            candidates.sort(
                key=lambda item: candidate_rank(target, item[1], threshold, fp_weight=fp_rank_weight),
                reverse=True,
            )
            centers = [row[0] for row in candidates[: max(int(elite_count), 1)]]
            fs *= float(sigma_decay)
            forearm_s *= float(sigma_decay)
        candidates.sort(
            key=lambda item: candidate_rank(target, item[1], threshold, fp_weight=fp_rank_weight),
            reverse=True,
        )
        pool = unique_candidates(candidates, max_candidates=max(int(pool_size), 1))
        pools[key] = pool
        best = pool[0]
        reports.append(
            {
                "keys": list(key),
                "frames": int(len(frames)),
                "pool_size": int(len(pool)),
                "best_source": str(best[2]),
                "best_static_f1": float(contact_f1(target, best[1], threshold)),
                "best_counts": list(contact_counts(target, best[1], threshold)),
            }
        )
    return pools, reports


def transition_score(
    *,
    previous_target: np.ndarray,
    target: np.ndarray,
    previous_activation: np.ndarray,
    activation: np.ndarray,
    previous_qpos: np.ndarray,
    qpos: np.ndarray,
    threshold: float,
    local_tp_weight: float,
    local_fp_weight: float,
    local_fn_weight: float,
    event_tp_weight: float,
    event_fp_weight: float,
    event_fn_weight: float,
    hold_fn_weight: float,
    change_weight: float,
) -> float:
    target_bool = np.asarray(target, dtype=np.float32)[:88] > threshold
    previous_target_bool = np.asarray(previous_target, dtype=np.float32)[:88] > threshold
    played_bool = np.asarray(activation, dtype=np.float32)[:88] > threshold
    previous_played_bool = np.asarray(previous_activation, dtype=np.float32)[:88] > threshold
    tp = int(np.logical_and(target_bool, played_bool).sum())
    fp = int(np.logical_and(~target_bool, played_bool).sum())
    fn = int(np.logical_and(target_bool, ~played_bool).sum())
    target_onset = target_bool & ~previous_target_bool
    played_onset = played_bool & ~previous_played_bool
    event_tp = int(np.logical_and(target_onset, played_onset).sum())
    event_fp = int(np.logical_and(~target_onset, played_onset).sum())
    event_fn = int(np.logical_and(target_onset, ~played_onset).sum())
    held_target = target_bool & previous_target_bool
    hold_fn = int(np.logical_and(held_target, ~played_bool).sum())
    change = float(np.linalg.norm(np.asarray(qpos, dtype=np.float32) - np.asarray(previous_qpos, dtype=np.float32)))
    return float(
        local_tp_weight * tp
        - local_fp_weight * fp
        - local_fn_weight * fn
        + event_tp_weight * event_tp
        - event_fp_weight * event_fp
        - event_fn_weight * event_fn
        - hold_fn_weight * hold_fn
        - change_weight * change
    )


def choose_sequence(
    *,
    target_keys: np.ndarray,
    masks: list[tuple[int, ...]],
    pools: dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray, str]]],
    threshold: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected_indices: list[int] = []
    backpointers: list[list[int]] = []
    scores: list[np.ndarray] = []
    first_pool = pools[masks[0]]
    first_scores = []
    zero = np.zeros((88,), dtype=np.float32)
    zero_qpos = first_pool[0][0]
    for qpos, activation, _source in first_pool:
        first_scores.append(
            transition_score(
                previous_target=zero,
                target=target_keys[0],
                previous_activation=zero,
                activation=activation,
                previous_qpos=zero_qpos,
                qpos=qpos,
                threshold=threshold,
                local_tp_weight=args.local_tp_weight,
                local_fp_weight=args.local_fp_weight,
                local_fn_weight=args.local_fn_weight,
                event_tp_weight=args.event_tp_weight,
                event_fp_weight=args.event_fp_weight,
                event_fn_weight=args.event_fn_weight,
                hold_fn_weight=args.hold_fn_weight,
                change_weight=args.change_weight,
            )
        )
    scores.append(np.asarray(first_scores, dtype=np.float64))
    backpointers.append([-1] * len(first_pool))

    for frame in range(1, len(masks)):
        previous_pool = pools[masks[frame - 1]]
        current_pool = pools[masks[frame]]
        current_scores = np.full((len(current_pool),), -np.inf, dtype=np.float64)
        current_back = np.zeros((len(current_pool),), dtype=np.int64)
        for cur_idx, (qpos, activation, _source) in enumerate(current_pool):
            best_score = -np.inf
            best_prev = 0
            for prev_idx, (prev_qpos, prev_activation, _prev_source) in enumerate(previous_pool):
                value = scores[-1][prev_idx] + transition_score(
                    previous_target=target_keys[frame - 1],
                    target=target_keys[frame],
                    previous_activation=prev_activation,
                    activation=activation,
                    previous_qpos=prev_qpos,
                    qpos=qpos,
                    threshold=threshold,
                    local_tp_weight=args.local_tp_weight,
                    local_fp_weight=args.local_fp_weight,
                    local_fn_weight=args.local_fn_weight,
                    event_tp_weight=args.event_tp_weight,
                    event_fp_weight=args.event_fp_weight,
                    event_fn_weight=args.event_fn_weight,
                    hold_fn_weight=args.hold_fn_weight,
                    change_weight=args.change_weight,
                )
                if value > best_score:
                    best_score = float(value)
                    best_prev = int(prev_idx)
            current_scores[cur_idx] = best_score
            current_back[cur_idx] = best_prev
        scores.append(current_scores)
        backpointers.append(current_back.astype(int).tolist())

    last = int(np.argmax(scores[-1]))
    selected_indices.append(last)
    for frame in range(len(masks) - 1, 0, -1):
        last = int(backpointers[frame][last])
        selected_indices.append(last)
    selected_indices.reverse()

    qpos_rows = []
    activation_rows = []
    sources = []
    for key, index in zip(masks, selected_indices):
        qpos, activation, source = pools[key][int(index)]
        qpos_rows.append(qpos)
        activation_rows.append(activation)
        sources.append(source)
    return (
        np.stack(qpos_rows, axis=0).astype(np.float32),
        np.stack(activation_rows, axis=0).astype(np.float32),
        sources,
    )


def proxy_event_counts(target_keys: np.ndarray, activations: np.ndarray, threshold: float) -> dict[str, int]:
    target = np.asarray(target_keys, dtype=np.float32)[:, :88] > threshold
    played = np.asarray(activations, dtype=np.float32)[:, :88] > threshold
    prev_target = np.zeros((88,), dtype=bool)
    prev_played = np.zeros((88,), dtype=bool)
    tp = fp = fn = 0
    for target_row, played_row in zip(target, played):
        target_on = target_row & ~prev_target
        played_on = played_row & ~prev_played
        tp += int(np.logical_and(target_on, played_on).sum())
        fp += int(np.logical_and(~target_on, played_on).sum())
        fn += int(np.logical_and(target_on, ~played_on).sum())
        prev_target = target_row
        prev_played = played_row
    return {"event_tp": int(tp), "event_fp": int(fp), "event_fn": int(fn)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mazurka: sequence-aware contact candidate selection for Impromptu.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--samples-per-mask", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--top-seeds", type=int, default=6)
    parser.add_argument("--finger-sigma", type=float, default=0.045)
    parser.add_argument("--forearm-sigma", type=float, default=0.030)
    parser.add_argument("--sigma-decay", type=float, default=0.55)
    parser.add_argument("--fp-rank-weight", type=float, default=0.06)
    parser.add_argument("--no-forearm", dest="include_forearm", action="store_false")
    parser.set_defaults(include_forearm=True)
    parser.add_argument("--local-tp-weight", type=float, default=1.00)
    parser.add_argument("--local-fp-weight", type=float, default=1.15)
    parser.add_argument("--local-fn-weight", type=float, default=1.10)
    parser.add_argument("--event-tp-weight", type=float, default=3.00)
    parser.add_argument("--event-fp-weight", type=float, default=2.80)
    parser.add_argument("--event-fn-weight", type=float, default=2.00)
    parser.add_argument("--hold-fn-weight", type=float, default=1.00)
    parser.add_argument("--change-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1732)
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
    config = BagatelleConfig(environment_name=str(args.environment_name), threshold=float(args.threshold), seed=0)
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        pools, pool_report = build_candidate_pools(
            kin=kin,
            target_keys=target_keys,
            control_qpos=control_qpos,
            threshold=float(args.threshold),
            rng=rng,
            samples_per_mask=int(args.samples_per_mask),
            iterations=int(args.iterations),
            elite_count=int(args.elite_count),
            pool_size=int(args.pool_size),
            top_seeds=int(args.top_seeds),
            finger_sigma=float(args.finger_sigma),
            forearm_sigma=float(args.forearm_sigma),
            sigma_decay=float(args.sigma_decay),
            fp_rank_weight=float(args.fp_rank_weight),
            include_forearm=bool(args.include_forearm),
        )
    masks = [mask_key(row, float(args.threshold)) for row in target_keys]
    control, proxy_activations, selected_sources = choose_sequence(
        target_keys=target_keys,
        masks=masks,
        pools=pools,
        threshold=float(args.threshold),
        args=args,
    )

    dense_qpos = np.repeat(control, substeps, axis=0).astype(np.float32)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    payload["planned_hand_joints"] = control.astype(np.float32)
    payload["planned_hand_joints_dense"] = dense_qpos
    payload["mazurka_proxy_activations"] = proxy_activations.astype(np.float32)
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
        "masks": int(len(pools)),
        "control_steps": int(control.shape[0]),
        "changed_control_frames": int(np.count_nonzero(np.linalg.norm(control - control_qpos, axis=1) > 1e-6)),
        "proxy_event_counts": proxy_event_counts(target_keys, proxy_activations, float(args.threshold)),
        "selected_sources_top": [
            {"source": source_name, "frames": int(count)}
            for source_name, count in sorted(
                ((source_name, selected_sources.count(source_name)) for source_name in set(selected_sources)),
                key=lambda item: (-item[1], item[0]),
            )[:20]
        ],
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "pool_report": pool_report[:300],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
