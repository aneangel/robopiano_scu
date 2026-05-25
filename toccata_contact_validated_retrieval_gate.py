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
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def mask_key(row: np.ndarray, threshold: float) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(np.asarray(row, dtype=np.float32)[:88] > float(threshold)).tolist())


def frame_f1(target: np.ndarray, played: np.ndarray, threshold: float) -> tuple[float, int, int, int]:
    goal = np.asarray(target, dtype=np.float32)[:88] > float(threshold)
    pred = np.asarray(played, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(goal, pred).sum())
    fp = int(np.logical_and(~goal, pred).sum())
    fn = int(np.logical_and(goal, ~pred).sum())
    denom = 2 * tp + fp + fn
    return (float(2 * tp / denom) if denom else 1.0, tp, fp, fn)


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def build_candidate_library(
    rp1m_root: Path,
    needed_masks: set[tuple[int, ...]],
    *,
    max_songs: int,
    demos_per_song: int,
    stride: int,
    threshold: float,
    per_mask_limit: int,
) -> dict[tuple[int, ...], list[np.ndarray]]:
    import zarr

    root = zarr.open(str(rp1m_root), mode="r")
    library: dict[tuple[int, ...], list[tuple[float, np.ndarray]]] = {key: [] for key in needed_masks}
    songs = [name for name in sorted(root.keys()) if hasattr(root[name], "keys")][: max(int(max_songs), 1)]
    for song in songs:
        group = root[song]
        if "goals" not in group or "hand_joints" not in group:
            continue
        states_available = "piano_states" in group
        demo_count = int(group["hand_joints"].shape[0])
        for demo_id in range(min(demo_count, int(demos_per_song))):
            goals = np.asarray(group["goals"][demo_id], dtype=np.float32)
            qpos = np.asarray(group["hand_joints"][demo_id], dtype=np.float32)
            states = np.asarray(group["piano_states"][demo_id], dtype=np.float32) if states_available else None
            steps = min(int(goals.shape[0]), int(qpos.shape[0]))
            for frame in range(0, steps, max(int(stride), 1)):
                key = mask_key(goals[frame], threshold)
                if key not in library:
                    continue
                quality = 1.0
                if states is not None:
                    quality = frame_f1(goals[frame], states[frame], threshold)[0]
                bucket = library[key]
                bucket.append((float(quality), qpos[frame].astype(np.float32)))
                bucket.sort(key=lambda item: item[0], reverse=True)
                if len(bucket) > int(per_mask_limit):
                    del bucket[int(per_mask_limit) :]
    return {key: [q for _quality, q in rows] for key, rows in library.items() if rows}


def interp_rows(anchor_x: np.ndarray, anchor_y: np.ndarray, out_x: np.ndarray) -> np.ndarray:
    anchor_x = np.asarray(anchor_x, dtype=np.float64).reshape(-1)
    anchor_y = np.asarray(anchor_y, dtype=np.float32)
    out_x = np.asarray(out_x, dtype=np.float64).reshape(-1)
    out = np.empty((out_x.size, anchor_y.shape[1]), dtype=np.float32)
    for col in range(anchor_y.shape[1]):
        out[:, col] = np.interp(out_x, anchor_x, anchor_y[:, col]).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Contact-validated RP1M retrieval for Impromptu hand-state postplanning.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rp1m-root", default="/WAVE/datasets/ccoelho_lab-jlanders/rp1m.zarr")
    parser.add_argument("--max-songs", type=int, default=80)
    parser.add_argument("--demos-per-song", type=int, default=2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--per-mask-limit", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-static-improvement", type=float, default=0.10)
    parser.add_argument("--max-fp-increase", type=int, default=0)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        target_keys = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
        baseline_control = np.asarray(data["planned_hand_joints"], dtype=np.float32)
        baseline_dense = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)
    control_steps = int(target_keys.shape[0])
    substeps = max(int(baseline_dense.shape[0] // max(control_steps, 1)), 1)
    needed = {mask_key(row, args.threshold) for row in target_keys}
    needed.discard(())
    library = build_candidate_library(
        Path(args.rp1m_root),
        needed,
        max_songs=args.max_songs,
        demos_per_song=args.demos_per_song,
        stride=args.stride,
        threshold=args.threshold,
        per_mask_limit=args.per_mask_limit,
    )
    config = BagatelleConfig(
        environment_name=str(args.environment_name),
        threshold=float(args.threshold),
        seed=int(args.seed),
        control_timestep=0.05,
    )
    selected = baseline_control.copy()
    used = np.zeros((control_steps,), dtype=bool)
    static_rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, ...], tuple[np.ndarray | None, dict[str, Any]]] = {}
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        for frame, goal in enumerate(target_keys):
            key = mask_key(goal, args.threshold)
            if not key:
                continue
            if key not in cache:
                baseline_activation = kin.activation_for_qpos(baseline_control[frame], settle_steps=1)
                base_score, base_tp, base_fp, base_fn = frame_f1(goal, baseline_activation, args.threshold)
                best_qpos: np.ndarray | None = None
                best = {
                    "static_f1": float(base_score),
                    "tp": int(base_tp),
                    "fp": int(base_fp),
                    "fn": int(base_fn),
                    "source": "baseline",
                }
                for idx, candidate in enumerate(library.get(key, [])):
                    activation = kin.activation_for_qpos(candidate, settle_steps=1)
                    score, tp, fp, fn = frame_f1(goal, activation, args.threshold)
                    improvement = float(score - base_score)
                    if improvement < float(args.min_static_improvement):
                        continue
                    if int(fp - base_fp) > int(args.max_fp_increase):
                        continue
                    rank = (float(score), int(tp), -int(fp), -int(fn))
                    best_rank = (float(best["static_f1"]), int(best["tp"]), -int(best["fp"]), -int(best["fn"]))
                    if rank > best_rank:
                        best_qpos = np.asarray(candidate, dtype=np.float32)
                        best = {
                            "static_f1": float(score),
                            "tp": int(tp),
                            "fp": int(fp),
                            "fn": int(fn),
                            "source": f"rp1m_candidate_{idx}",
                            "baseline_static_f1": float(base_score),
                            "baseline_tp": int(base_tp),
                            "baseline_fp": int(base_fp),
                            "baseline_fn": int(base_fn),
                            "candidates": int(len(library.get(key, []))),
                        }
                cache[key] = (best_qpos, best)
            qpos, row = cache[key]
            if qpos is not None:
                selected[frame] = qpos
                used[frame] = True
            if frame < 20 or qpos is not None:
                static_rows.append({"frame": int(frame), "keys": list(key), **row})

    control_x = np.arange(control_steps, dtype=np.float64)
    dense_x = np.arange(control_steps * substeps, dtype=np.float64) / float(substeps)
    dense_qpos = interp_rows(control_x, selected, dense_x)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    atomic_save_npz(
        out / "trajectory.npz",
        target_keys=target_keys,
        planned_hand_joints=selected.astype(np.float32),
        planned_hand_joints_dense=dense_qpos.astype(np.float32),
        retrieval_used=used.astype(np.float32),
    )
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
        "library_masks": int(len(library)),
        "needed_masks": int(len(needed)),
        "retrieved_control_frames": int(used.sum()),
        "control_steps": int(control_steps),
        "retrieval_fraction": float(used.mean()),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "static_rows": static_rows[:200],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
