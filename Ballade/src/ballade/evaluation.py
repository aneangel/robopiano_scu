from __future__ import annotations

from typing import Any

import numpy as np

from ballade.costs import action_saturation_cost


def key_metrics(target: np.ndarray, played: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    target_active = np.asarray(target, dtype=np.float32)[:, :88] > float(threshold)
    played_active = np.asarray(played, dtype=np.float32)[:, :88] > float(threshold)
    steps = min(target_active.shape[0], played_active.shape[0])
    if steps <= 0:
        return {"goal_key_precision": 0.0, "goal_key_recall": 0.0, "goal_key_f1": 0.0, "mispress_rate": 0.0}
    target_active = target_active[:steps]
    played_active = played_active[:steps]
    tp = int(np.logical_and(target_active, played_active).sum())
    fp = int(np.logical_and(~target_active, played_active).sum())
    fn = int(np.logical_and(target_active, ~played_active).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    mispress = np.logical_and(~target_active, played_active).any(axis=1)
    return {
        "goal_key_precision": precision,
        "goal_key_recall": recall,
        "goal_key_f1": f1,
        "mispress_rate": float(mispress.mean()) if mispress.size else 0.0,
    }


def action_metrics(actions: np.ndarray) -> dict[str, float]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.size == 0:
        return {"action_saturation_fraction": 0.0, "action_delta_p95": 0.0}
    saturated = np.abs(arr) >= 0.98
    if arr.shape[0] <= 1:
        delta_p95 = 0.0
    else:
        delta = np.linalg.norm(np.diff(arr.reshape(arr.shape[0], -1), axis=0), axis=1)
        delta_p95 = float(np.percentile(delta, 95))
    return {
        "action_saturation_fraction": float(saturated.mean()),
        "action_saturation_cost": action_saturation_cost(arr),
        "action_delta_p95": delta_p95,
    }


def hand_l2_metrics(observed: np.ndarray, target: np.ndarray) -> dict[str, float]:
    obs = np.asarray(observed, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    steps = min(obs.shape[0], tgt.shape[0])
    width = min(obs.shape[-1], tgt.shape[-1]) if obs.ndim >= 2 and tgt.ndim >= 2 else 0
    if steps <= 0 or width <= 0:
        return {"hand_qpos_l2_mean": float("nan"), "hand_qpos_l2_p95": float("nan")}
    l2 = np.linalg.norm(obs[:steps, :width] - tgt[:steps, :width], axis=1)
    return {"hand_qpos_l2_mean": float(l2.mean()), "hand_qpos_l2_p95": float(np.percentile(l2, 95))}


def summarize_rollout(
    *,
    dense_goal_mask: np.ndarray,
    dense_played: np.ndarray,
    dense_hand: np.ndarray,
    dense_target_hand: np.ndarray,
    actions: np.ndarray,
    search_used: np.ndarray | None = None,
    terminated: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary.update(key_metrics(dense_goal_mask, dense_played))
    summary.update(hand_l2_metrics(dense_hand, dense_target_hand))
    summary.update(action_metrics(actions))
    summary["audio_note_event_count"] = int(np.logical_and(dense_played[:-1] <= 0.5, dense_played[1:] > 0.5).sum()) if dense_played.shape[0] > 1 else 0
    summary["search_usage_fraction"] = 0.0 if search_used is None or search_used.size == 0 else float(np.mean(search_used))
    summary["closed_loop_terminated"] = bool(terminated)
    return summary
