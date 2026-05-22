from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nocturne.events import press_frame_indices
from nocturne.io import ensure_dir, save_json, save_npz
from nocturne.schema import NoteEvent


DEFAULT_GOAL_LOOKAHEAD = (0, 1, 5, 10)
DEFAULT_Q_LOOKAHEAD = (0, 1, 5)


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T, D], got {arr.shape}")
    if arr.shape[0] <= 1:
        return np.zeros_like(arr)
    return np.gradient(arr, max(float(dt), 1e-6), axis=0, edge_order=1).astype(np.float32)


def build_features_for_trajectory(
    *,
    q: np.ndarray,
    qdot: np.ndarray,
    actions: np.ndarray,
    goals: np.ndarray,
    q_ref: np.ndarray | None = None,
    previous_actions: np.ndarray | None = None,
    press_frames: np.ndarray | None = None,
    goal_lookahead: tuple[int, ...] = DEFAULT_GOAL_LOOKAHEAD,
    q_lookahead: tuple[int, ...] = DEFAULT_Q_LOOKAHEAD,
) -> tuple[np.ndarray, list[str]]:
    q = np.asarray(q, dtype=np.float32)
    qdot = np.asarray(qdot, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.float32)[:, :88]
    q_ref = q if q_ref is None else np.asarray(q_ref, dtype=np.float32)
    previous_actions = _previous_actions(actions) if previous_actions is None else np.asarray(previous_actions, dtype=np.float32)
    total = int(min(q.shape[0], qdot.shape[0], actions.shape[0], goals.shape[0], q_ref.shape[0], previous_actions.shape[0]))
    press_mask = _press_mask(goals[:total])
    if press_frames is None:
        press_frames = np.flatnonzero(press_mask).astype(np.int64)
    time_to_next = _time_to_next_press(total, np.asarray(press_frames, dtype=np.int64))
    active_mask = np.any(goals[:total] > 0.5, axis=1).astype(np.float32)

    rows = []
    for t in range(total):
        parts = [
            q[t],
            qdot[t],
            previous_actions[t],
            np.asarray([time_to_next[t], press_mask[t], active_mask[t]], dtype=np.float32),
        ]
        for step in goal_lookahead:
            parts.append(goals[_bounded(t + int(step), total)])
        for step in q_lookahead:
            parts.append(q_ref[_bounded(t + int(step), total)])
        rows.append(np.concatenate(parts).astype(np.float32))
    names = _feature_names(q.shape[1], qdot.shape[1], actions.shape[1], goal_lookahead, q_lookahead)
    return np.stack(rows, axis=0).astype(np.float32), names


def build_controller_dataset(
    trajectory_npz: str | Path,
    output_root: str | Path,
    *,
    press_window_radius: int = 2,
    press_weight: float = 5.0,
) -> dict[str, Any]:
    out = ensure_dir(output_root)
    with np.load(Path(trajectory_npz), allow_pickle=False) as data:
        q = np.asarray(data["hand_joints"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        goals = np.asarray(data["goals"], dtype=np.float32)
        fingertips = np.asarray(data["hand_fingertips"], dtype=np.float32)
        dt = float(np.asarray(data["dt"]).reshape(())) if "dt" in data else 0.05
        stored_press = np.asarray(data["press_frame_indices"], dtype=np.int64) if "press_frame_indices" in data else None
    qdot = finite_difference(q, dt)
    features, names = build_features_for_trajectory(
        q=q,
        qdot=qdot,
        actions=actions,
        goals=goals,
        q_ref=q,
        press_frames=stored_press,
    )
    weights = np.ones((features.shape[0],), dtype=np.float32)
    press_mask = np.zeros_like(weights, dtype=bool)
    frames = stored_press if stored_press is not None else np.flatnonzero(_press_mask(goals[:, :88]))
    for frame in np.asarray(frames, dtype=np.int64).reshape(-1):
        start = max(int(frame) - int(press_window_radius), 0)
        end = min(int(frame) + int(press_window_radius) + 1, weights.shape[0])
        press_mask[start:end] = True
    weights[press_mask] = float(press_weight)

    split = _deterministic_split(features.shape[0])
    save_npz(
        out / "dataset.npz",
        features=features,
        actions=actions[: features.shape[0]].astype(np.float32),
        weights=weights,
        split=split,
        q=q[: features.shape[0]].astype(np.float32),
        qdot=qdot[: features.shape[0]].astype(np.float32),
        goals=goals[: features.shape[0]].astype(np.float32),
        fingertips=fingertips[: features.shape[0]].astype(np.float32),
        dt=np.asarray(dt, dtype=np.float32),
    )
    summary = {
        "trajectory_npz": str(Path(trajectory_npz)),
        "dataset_npz": str(out / "dataset.npz"),
        "num_frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "action_dim": int(actions.shape[1]),
        "feature_names": names,
        "privileged_exclusions": ["piano_states", "source_demo_id_per_frame", "segment_id_per_frame"],
        "train_frames": int(np.count_nonzero(split == 0)),
        "val_frames": int(np.count_nonzero(split == 1)),
    }
    save_json(out / "dataset_summary.json", summary)
    (out / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    return summary


def build_online_feature(
    *,
    q: np.ndarray,
    qdot: np.ndarray,
    previous_action: np.ndarray,
    q_ref: np.ndarray,
    goals: np.ndarray,
    t: int,
) -> np.ndarray:
    actions_stub = np.zeros((q_ref.shape[0], previous_action.reshape(-1).shape[0]), dtype=np.float32)
    prev = np.repeat(np.asarray(previous_action, dtype=np.float32).reshape(1, -1), q_ref.shape[0], axis=0)
    q_live = np.asarray(q_ref, dtype=np.float32).copy()
    qdot_live = finite_difference(q_live, 0.05)
    index = _bounded(int(t), q_live.shape[0])
    q_live[index] = np.asarray(q, dtype=np.float32).reshape(-1)
    qdot_live[index] = np.asarray(qdot, dtype=np.float32).reshape(-1)
    features, _ = build_features_for_trajectory(
        q=q_live,
        qdot=qdot_live,
        actions=actions_stub,
        goals=goals,
        q_ref=q_ref,
        previous_actions=prev,
    )
    return features[index]


def _feature_names(
    q_dim: int,
    qdot_dim: int,
    action_dim: int,
    goal_lookahead: tuple[int, ...],
    q_lookahead: tuple[int, ...],
) -> list[str]:
    names = [f"q_{i}" for i in range(q_dim)]
    names.extend(f"qdot_{i}" for i in range(qdot_dim))
    names.extend(f"previous_action_{i}" for i in range(action_dim))
    names.extend(["time_to_next_press", "is_press_frame", "is_active_goal_frame"])
    for step in goal_lookahead:
        names.extend(f"goal_t_plus_{step}_key_{i}" for i in range(88))
    for step in q_lookahead:
        names.extend(f"q_ref_t_plus_{step}_{i}" for i in range(q_dim))
    return names


def _previous_actions(actions: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(actions, dtype=np.float32)
    if actions.shape[0] > 1:
        prev[1:] = actions[:-1]
    return prev


def _press_mask(goals: np.ndarray) -> np.ndarray:
    active = np.asarray(goals, dtype=np.float32)[:, :88] > 0.5
    previous = np.zeros((active.shape[1],), dtype=bool)
    mask = np.zeros((active.shape[0],), dtype=np.float32)
    for frame, row in enumerate(active):
        if bool(np.any(row & ~previous)):
            mask[frame] = 1.0
        previous = row
    return mask


def _time_to_next_press(total: int, press_frames: np.ndarray) -> np.ndarray:
    out = np.ones((int(total),), dtype=np.float32)
    frames = sorted(int(frame) for frame in np.asarray(press_frames, dtype=np.int64).reshape(-1).tolist())
    cursor = 0
    for t in range(int(total)):
        while cursor < len(frames) and frames[cursor] < t:
            cursor += 1
        if cursor < len(frames):
            out[t] = float(frames[cursor] - t) / max(float(total), 1.0)
        else:
            out[t] = 1.0
    return out


def _deterministic_split(total: int) -> np.ndarray:
    split = np.zeros((int(total),), dtype=np.int64)
    if total >= 5:
        split[np.arange(total) % 5 == 0] = 1
    return split


def _bounded(index: int, total: int) -> int:
    return int(np.clip(index, 0, max(int(total) - 1, 0)))
