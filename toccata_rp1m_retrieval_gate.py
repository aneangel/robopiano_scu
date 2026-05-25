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
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "partita" / "src",
):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def _mask_key(row: np.ndarray, threshold: float) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(np.asarray(row, dtype=np.float32)[:88] > float(threshold)).tolist())


def _frame_f1(goal: np.ndarray, state: np.ndarray, threshold: float) -> float:
    g = np.asarray(goal, dtype=np.float32)[:88] > float(threshold)
    s = np.asarray(state, dtype=np.float32)[:88] > float(threshold)
    tp = int(np.logical_and(g, s).sum())
    fp = int(np.logical_and(~g, s).sum())
    fn = int(np.logical_and(g, ~s).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 1.0


def build_exact_library(
    rp1m_root: Path,
    *,
    max_songs: int,
    demos_per_song: int,
    stride: int,
    threshold: float,
    min_recorded_frame_f1: float,
    max_polyphony: int,
) -> dict[tuple[int, ...], tuple[np.ndarray, float]]:
    import zarr

    root = zarr.open(str(rp1m_root), mode="r")
    library: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    songs = [name for name in sorted(root.keys()) if hasattr(root[name], "keys")][: max(int(max_songs), 1)]
    for song in songs:
        group = root[song]
        if "goals" not in group or "hand_joints" not in group:
            continue
        demo_count = int(group["hand_joints"].shape[0])
        for demo_id in range(min(demo_count, int(demos_per_song))):
            goals = np.asarray(group["goals"][demo_id], dtype=np.float32)
            qpos = np.asarray(group["hand_joints"][demo_id], dtype=np.float32)
            states = np.asarray(group["piano_states"][demo_id], dtype=np.float32) if "piano_states" in group else None
            steps = min(int(goals.shape[0]), int(qpos.shape[0]))
            for frame in range(0, steps, max(int(stride), 1)):
                key = _mask_key(goals[frame], threshold)
                if not key or len(key) > int(max_polyphony):
                    continue
                quality = 1.0 if states is None else _frame_f1(goals[frame], states[frame], threshold)
                if quality < float(min_recorded_frame_f1):
                    continue
                current = library.get(key)
                if current is None or quality > current[1]:
                    library[key] = (qpos[frame].astype(np.float32), float(quality))
    return library


def _interp_rows(anchor_x: np.ndarray, anchor_y: np.ndarray, out_x: np.ndarray) -> np.ndarray:
    anchor_x = np.asarray(anchor_x, dtype=np.float64).reshape(-1)
    anchor_y = np.asarray(anchor_y, dtype=np.float32)
    out_x = np.asarray(out_x, dtype=np.float64).reshape(-1)
    out = np.empty((out_x.size, anchor_y.shape[1]), dtype=np.float32)
    for col in range(anchor_y.shape[1]):
        out[:, col] = np.interp(out_x, anchor_x, anchor_y[:, col]).astype(np.float32)
    return out


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype exact-mask RP1M retrieval as an Impromptu hand-state postplanner.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rp1m-root", default="/WAVE/datasets/ccoelho_lab-jlanders/rp1m.zarr")
    parser.add_argument("--max-songs", type=int, default=120)
    parser.add_argument("--demos-per-song", type=int, default=2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-recorded-frame-f1", type=float, default=0.95)
    parser.add_argument("--max-polyphony", type=int, default=5)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--hand-anchor-y-offset", type=float, default=None)
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    library = build_exact_library(
        Path(args.rp1m_root),
        max_songs=args.max_songs,
        demos_per_song=args.demos_per_song,
        stride=args.stride,
        threshold=args.threshold,
        min_recorded_frame_f1=args.min_recorded_frame_f1,
        max_polyphony=args.max_polyphony,
    )
    with np.load(source, allow_pickle=False) as data:
        target_keys = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
        baseline_control = np.asarray(data["planned_hand_joints"], dtype=np.float32)
        baseline_dense = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)
    control_steps = int(target_keys.shape[0])
    substeps = max(int(baseline_dense.shape[0] // max(control_steps, 1)), 1)
    qpos = baseline_control.copy()
    used = np.zeros((control_steps,), dtype=bool)
    qualities = np.zeros((control_steps,), dtype=np.float32)
    for frame, row in enumerate(target_keys):
        key = _mask_key(row, args.threshold)
        item = library.get(key)
        if item is None:
            continue
        qpos[frame] = item[0]
        used[frame] = True
        qualities[frame] = item[1]
    dense_x = np.arange(control_steps * substeps, dtype=np.float64) / float(substeps)
    control_x = np.arange(control_steps, dtype=np.float64)
    dense_qpos = _interp_rows(control_x, qpos, dense_x)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    atomic_save_npz(
        out / "trajectory.npz",
        target_keys=target_keys,
        planned_hand_joints=qpos.astype(np.float32),
        planned_hand_joints_dense=dense_qpos.astype(np.float32),
        retrieval_used=used.astype(np.float32),
        retrieval_quality=qualities.astype(np.float32),
    )
    trajectory = make_rp1m_trajectory_from_arrays(
        song_key=args.environment_name,
        demo_id=0,
        actions=np.zeros((dense_qpos.shape[0], 39), dtype=np.float32),
        goals=dense_goals,
        hand_joints=dense_qpos,
        environment_name=args.environment_name,
    )
    config = RolloutConfig(
        mode="hand_state",
        dataset_timestep=dense_dt,
        simulation_timestep=dense_dt,
        hand_anchor_y_offset=args.hand_anchor_y_offset,
        hand_state_action_source="zero",
        restore_initial_hand=True,
        set_hand_qvel=False,
        threshold=float(args.threshold),
        render_mp4=False,
        render_audio=False,
    )
    summary = simulate_rp1m_rollout(trajectory, config, out / "rp1m_sim")
    with np.load(summary["rollout_npz"], allow_pickle=False) as rollout:
        played = np.asarray(rollout["source_played_piano"], dtype=np.float32)
        goals = np.asarray(rollout["goals"], dtype=np.float32)
    score = score_rollout(
        target_keys=goals,
        played_keys=played,
        dt=dense_dt,
        threshold=float(args.threshold),
        timing_tolerance_s=0.15,
    )
    result = {
        "source_npz": str(source),
        "output_dir": str(out),
        "library_size": int(len(library)),
        "retrieved_control_frames": int(used.sum()),
        "control_steps": int(control_steps),
        "retrieval_fraction": float(used.mean()),
        "mean_retrieval_quality": float(qualities[used].mean()) if bool(used.any()) else 0.0,
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "hand_anchor_y_offset": args.hand_anchor_y_offset,
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
