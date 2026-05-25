from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_PREFIX_STEPS: tuple[int, ...] = (10, 25, 50, 100, 150, 200, 300)


def compute_source_hand_errors(
    *,
    sim_hand_joints: np.ndarray,
    target_hand_joints: np.ndarray,
    dt: float = 0.05,
    target_offset: int = 1,
) -> dict[str, np.ndarray]:
    """Measure per-source-step drift from the planner hand-state trajectory.

    `sim_hand_joints[i]` is the hand state captured after executing controller
    interval `i`. The planner endpoint for that interval is `target[i + 1]` by
    default, matching the closed-loop summary metric already used by Fugue and
    Allegro.
    """

    sim = np.asarray(sim_hand_joints, dtype=np.float32)
    target = np.asarray(target_hand_joints, dtype=np.float32)
    if sim.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected sim and target with shape [T, D], got {sim.shape} and {target.shape}")
    if sim.shape[1] <= 0 or target.shape[1] <= 0:
        raise ValueError("Hand-state arrays must have a nonzero joint dimension")
    offset = int(target_offset)
    if offset < 0:
        raise ValueError("target_offset must be nonnegative")
    steps = min(int(sim.shape[0]), max(int(target.shape[0]) - offset, 0))
    width = min(int(sim.shape[1]), int(target.shape[1]))
    if steps <= 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            "source_step": empty.astype(np.int64),
            "elapsed_s": empty,
            "target_step": empty.astype(np.int64),
            "hand_l2": empty,
            "hand_rmse": empty,
            "hand_max_abs": empty,
            "right_l2": empty,
            "left_l2": empty,
        }
    diff = sim[:steps, :width] - target[offset : offset + steps, :width]
    half = width // 2
    right = diff[:, :half] if half > 0 else diff[:, :0]
    left = diff[:, half:] if half < width else diff[:, :0]
    return {
        "source_step": np.arange(steps, dtype=np.int64),
        "elapsed_s": (np.arange(steps, dtype=np.float32) + 1.0) * float(dt),
        "target_step": np.arange(offset, offset + steps, dtype=np.int64),
        "hand_l2": np.linalg.norm(diff, axis=1).astype(np.float32),
        "hand_rmse": np.sqrt(np.mean(np.square(diff), axis=1)).astype(np.float32),
        "hand_max_abs": np.max(np.abs(diff), axis=1).astype(np.float32),
        "right_l2": np.linalg.norm(right, axis=1).astype(np.float32)
        if right.shape[1] > 0
        else np.zeros((steps,), dtype=np.float32),
        "left_l2": np.linalg.norm(left, axis=1).astype(np.float32)
        if left.shape[1] > 0
        else np.zeros((steps,), dtype=np.float32),
    }


def summarize_error_prefixes(
    errors: dict[str, np.ndarray],
    *,
    prefixes: Iterable[int] = DEFAULT_PREFIX_STEPS,
) -> list[dict[str, Any]]:
    hand_l2 = np.asarray(errors["hand_l2"], dtype=np.float32)
    hand_rmse = np.asarray(errors["hand_rmse"], dtype=np.float32)
    hand_max_abs = np.asarray(errors["hand_max_abs"], dtype=np.float32)
    elapsed = np.asarray(errors["elapsed_s"], dtype=np.float32)
    total = int(hand_l2.shape[0])
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        count = min(int(prefix), total)
        if count <= 0:
            continue
        rows.append(_summary_row(count, hand_l2=hand_l2, hand_rmse=hand_rmse, hand_max_abs=hand_max_abs, elapsed=elapsed))
    if total > 0 and all(int(row["source_steps"]) != total for row in rows):
        rows.append(_summary_row(total, hand_l2=hand_l2, hand_rmse=hand_rmse, hand_max_abs=hand_max_abs, elapsed=elapsed))
    return rows


def _summary_row(
    count: int,
    *,
    hand_l2: np.ndarray,
    hand_rmse: np.ndarray,
    hand_max_abs: np.ndarray,
    elapsed: np.ndarray,
) -> dict[str, Any]:
    values = hand_l2[:count]
    return {
        "source_steps": int(count),
        "elapsed_s": float(elapsed[count - 1]) if elapsed.size else 0.0,
        "hand_l2_mean": float(values.mean()),
        "hand_l2_median": float(np.median(values)),
        "hand_l2_p95": float(np.percentile(values, 95)),
        "hand_l2_max": float(values.max()),
        "hand_l2_final": float(values[-1]),
        "hand_rmse_mean": float(hand_rmse[:count].mean()),
        "hand_max_abs_p95": float(np.percentile(hand_max_abs[:count], 95)),
        "hand_l2_drift_per_s": _linear_slope(elapsed[:count], values),
    }


def write_error_csv(path: str | Path, errors: dict[str, np.ndarray]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_step",
        "elapsed_s",
        "target_step",
        "hand_l2",
        "hand_rmse",
        "hand_max_abs",
        "right_l2",
        "left_l2",
    ]
    count = int(np.asarray(errors["hand_l2"]).shape[0])
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx in range(count):
            writer.writerow({field: _jsonable_scalar(errors[field][idx]) for field in fields})
    return out


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_rollout_hand_arrays(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(npz_path), allow_pickle=False)
    if "sim_hand_joints" not in data or "target_hand_joints" not in data:
        raise KeyError(f"{npz_path} must contain sim_hand_joints and target_hand_joints")
    return np.asarray(data["sim_hand_joints"], dtype=np.float32), np.asarray(data["target_hand_joints"], dtype=np.float32)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return None
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    centered = x64 - x64.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return None
    return float(np.dot(centered, y64 - y64.mean()) / denom)


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
