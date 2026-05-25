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


RIGHT_SLICE = slice(0, 23)
LEFT_SLICE = slice(23, 46)


def _keys(row: np.ndarray, threshold: float) -> np.ndarray:
    return np.flatnonzero(np.asarray(row, dtype=np.float32)[:88] > float(threshold)).astype(np.int32)


def _side_key(mask: np.ndarray, *, split_key: int, side: str) -> tuple[int, ...]:
    keys = np.asarray(mask, dtype=np.int32).reshape(-1)
    if side == "left":
        values = keys[keys < int(split_key)]
    elif side == "right":
        values = keys[keys >= int(split_key)]
    else:
        raise ValueError(f"Unknown side: {side}")
    return tuple(int(v) for v in values.tolist())


def _side_frame_f1(goal: np.ndarray, state: np.ndarray, *, threshold: float, split_key: int, side: str) -> float:
    g = np.asarray(goal, dtype=np.float32)[:88] > float(threshold)
    s = np.asarray(state, dtype=np.float32)[:88] > float(threshold)
    if side == "left":
        region = np.arange(88) < int(split_key)
    else:
        region = np.arange(88) >= int(split_key)
    g = g[region]
    s = s[region]
    tp = int(np.logical_and(g, s).sum())
    fp = int(np.logical_and(~g, s).sum())
    fn = int(np.logical_and(g, ~s).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 1.0


def _event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _choose_better(current: tuple[np.ndarray, float, int] | None, candidate: tuple[np.ndarray, float, int]) -> tuple[np.ndarray, float, int]:
    if current is None:
        return candidate
    if candidate[1] > current[1]:
        return candidate
    if np.isclose(candidate[1], current[1]) and candidate[2] > current[2]:
        return candidate
    return current


def build_side_libraries(
    rp1m_root: Path,
    *,
    max_songs: int,
    demos_per_song: int,
    stride: int,
    threshold: float,
    split_key: int,
    min_side_frame_f1: float,
    max_side_polyphony: int,
) -> tuple[dict[tuple[int, ...], tuple[np.ndarray, float, int]], dict[tuple[int, ...], tuple[np.ndarray, float, int]]]:
    import zarr

    root = zarr.open(str(rp1m_root), mode="r")
    left: dict[tuple[int, ...], tuple[np.ndarray, float, int]] = {}
    right: dict[tuple[int, ...], tuple[np.ndarray, float, int]] = {}
    songs = [name for name in sorted(root.keys()) if hasattr(root[name], "keys")][: max(int(max_songs), 1)]
    for song in songs:
        group = root[song]
        if "goals" not in group or "hand_joints" not in group:
            continue
        states_available = "piano_states" in group
        demo_count = int(group["hand_joints"].shape[0])
        for demo_id in range(min(demo_count, max(int(demos_per_song), 1))):
            goals = np.asarray(group["goals"][demo_id], dtype=np.float32)
            qpos = np.asarray(group["hand_joints"][demo_id], dtype=np.float32)
            states = np.asarray(group["piano_states"][demo_id], dtype=np.float32) if states_available else None
            steps = min(int(goals.shape[0]), int(qpos.shape[0]))
            for frame in range(0, steps, max(int(stride), 1)):
                keys = _keys(goals[frame], threshold)
                for side, library, hand_slice in (("left", left, LEFT_SLICE), ("right", right, RIGHT_SLICE)):
                    key = _side_key(keys, split_key=split_key, side=side)
                    if len(key) > int(max_side_polyphony):
                        continue
                    quality = (
                        1.0
                        if states is None
                        else _side_frame_f1(goals[frame], states[frame], threshold=threshold, split_key=split_key, side=side)
                    )
                    if quality < float(min_side_frame_f1):
                        continue
                    hand = qpos[frame, hand_slice].astype(np.float32).copy()
                    current = library.get(key)
                    library[key] = _choose_better(current, (hand, float(quality), 1))
    return left, right


def make_dense(qpos: np.ndarray, *, substeps: int, mode: str) -> np.ndarray:
    rows = np.asarray(qpos, dtype=np.float32)
    sub = max(int(substeps), 1)
    if mode == "zoh":
        return np.repeat(rows, sub, axis=0).astype(np.float32)
    if mode != "linear":
        raise ValueError(f"dense_mode must be zoh or linear, got {mode!r}")
    control_x = np.arange(rows.shape[0], dtype=np.float64)
    dense_x = np.arange(rows.shape[0] * sub, dtype=np.float64) / float(sub)
    out = np.empty((dense_x.size, rows.shape[1]), dtype=np.float32)
    for col in range(rows.shape[1]):
        out[:, col] = np.interp(dense_x, control_x, rows[:, col]).astype(np.float32)
    return out


def run_variant(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    left_library, right_library = build_side_libraries(
        Path(args.rp1m_root),
        max_songs=args.max_songs,
        demos_per_song=args.demos_per_song,
        stride=args.stride,
        threshold=args.threshold,
        split_key=args.split_key,
        min_side_frame_f1=args.min_side_frame_f1,
        max_side_polyphony=args.max_side_polyphony,
    )
    with np.load(source, allow_pickle=False) as data:
        target_keys = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
        control = np.asarray(data["planned_hand_joints"], dtype=np.float32)
        base_dense = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)
    control_steps = int(target_keys.shape[0])
    substeps = max(int(base_dense.shape[0] // max(control_steps, 1)), 1)
    qpos = control.copy()
    used_left = np.zeros((control_steps,), dtype=bool)
    used_right = np.zeros((control_steps,), dtype=bool)
    left_quality = np.zeros((control_steps,), dtype=np.float32)
    right_quality = np.zeros((control_steps,), dtype=np.float32)
    for frame, row in enumerate(target_keys):
        keys = _keys(row, args.threshold)
        left_key = _side_key(keys, split_key=args.split_key, side="left")
        right_key = _side_key(keys, split_key=args.split_key, side="right")
        if left_key or bool(args.replace_empty_side):
            item = left_library.get(left_key)
            if item is not None:
                qpos[frame, LEFT_SLICE] = item[0]
                left_quality[frame] = item[1]
                used_left[frame] = True
        if right_key or bool(args.replace_empty_side):
            item = right_library.get(right_key)
            if item is not None:
                qpos[frame, RIGHT_SLICE] = item[0]
                right_quality[frame] = item[1]
                used_right[frame] = True
    dense_qpos = make_dense(qpos, substeps=substeps, mode=str(args.dense_mode))
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    atomic_save_npz(
        out / "trajectory.npz",
        target_keys=target_keys,
        planned_hand_joints=qpos.astype(np.float32),
        planned_hand_joints_dense=dense_qpos.astype(np.float32),
        scherzo_left_used=used_left.astype(np.float32),
        scherzo_right_used=used_right.astype(np.float32),
        scherzo_left_quality=left_quality.astype(np.float32),
        scherzo_right_quality=right_quality.astype(np.float32),
    )
    trajectory = make_rp1m_trajectory_from_arrays(
        song_key=args.environment_name,
        demo_id=0,
        actions=np.zeros((dense_qpos.shape[0], 39), dtype=np.float32),
        goals=dense_goals,
        hand_joints=dense_qpos,
        environment_name=args.environment_name,
    )
    summary = simulate_rp1m_rollout(
        trajectory,
        RolloutConfig(
            mode="hand_state",
            dataset_timestep=float(dense_dt),
            simulation_timestep=float(dense_dt),
            hand_anchor_y_offset=args.hand_anchor_y_offset,
            hand_state_action_source="zero",
            restore_initial_hand=True,
            set_hand_qvel=False,
            threshold=float(args.threshold),
            render_mp4=False,
            render_audio=False,
        ),
        out / "rp1m_sim",
    )
    with np.load(summary["rollout_npz"], allow_pickle=False) as rollout:
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
        "left_library_size": int(len(left_library)),
        "right_library_size": int(len(right_library)),
        "split_key": int(args.split_key),
        "dense_mode": str(args.dense_mode),
        "replace_empty_side": bool(args.replace_empty_side),
        "left_retrieved_control_frames": int(used_left.sum()),
        "right_retrieved_control_frames": int(used_right.sum()),
        "control_steps": int(control_steps),
        "left_retrieval_fraction": float(used_left.mean()) if used_left.size else 0.0,
        "right_retrieval_fraction": float(used_right.mean()) if used_right.size else 0.0,
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(_event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((summary.get("against_goals") or {}).get("key_f1", 0.0)),
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scherzo: split-hand RP1M retrieval postplanner for Impromptu.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rp1m-root", default="/WAVE/datasets/ccoelho_lab-jlanders/rp1m.zarr")
    parser.add_argument("--max-songs", type=int, default=120)
    parser.add_argument("--demos-per-song", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-key", type=int, default=44)
    parser.add_argument("--min-side-frame-f1", type=float, default=0.95)
    parser.add_argument("--max-side-polyphony", type=int, default=5)
    parser.add_argument("--dense-mode", choices=("zoh", "linear"), default="zoh")
    parser.add_argument("--replace-empty-side", action="store_true")
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--hand-anchor-y-offset", type=float, default=None)
    args = parser.parse_args()
    run_variant(args)


if __name__ == "__main__":
    main()
