#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
for _path in (REPO_ROOT, REPO_ROOT / "Intermezzo" / "src", REPO_ROOT / "partita" / "src"):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


RIGHT_HAND = np.arange(0, 23, dtype=np.int64)
LEFT_HAND = np.arange(23, 46, dtype=np.int64)
FULL_HANDS = np.arange(46, dtype=np.int64)


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
        int(score["matched"]),
        -int(score["mispresses"]),
        float(score["frame_f1"]),
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(value.strip())
    return path.parent.name, path


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
        score = score_rollout(
            target_keys=target,
            played_keys=played,
            dt=float(dense_dt),
            threshold=float(threshold),
            timing_tolerance_s=0.15,
        )
        return compact_score(score), score
    except Exception as exc:
        bad = {
            "event_f1": -1.0,
            "frame_f1": -1.0,
            "matched": 0,
            "target": 0,
            "played": 10**9,
            "mispresses": 10**9,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return bad, {"error": bad["error"], "missed_events": [], "mispress_events": []}


def indices_for_scope(scope: str, key: int, split_key: int) -> np.ndarray:
    if scope == "full":
        return FULL_HANDS
    target_hand = LEFT_HAND if int(key) < int(split_key) else RIGHT_HAND
    other_hand = RIGHT_HAND if int(key) < int(split_key) else LEFT_HAND
    if scope == "target_hand":
        return target_hand
    if scope == "other_hand":
        return other_hand
    if scope == "both_hands":
        return FULL_HANDS
    raise ValueError(f"unknown scope {scope!r}")


def build_event_windows(
    detail: dict[str, Any],
    *,
    event_types: set[str],
    pre_dense_frames: int,
    post_dense_frames: int,
    dense_total: int,
    max_events: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if "mispress" in event_types:
        for item in detail.get("mispress_events", []):
            events.append({"type": "mispress", "frame": int(item.get("frame", 0)), "key": int(item.get("key", -1))})
    if "missed" in event_types:
        for item in detail.get("missed_events", []):
            events.append({"type": "missed", "frame": int(item.get("frame", 0)), "key": int(item.get("key", -1))})
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for item in sorted(events, key=lambda row: (int(row["frame"]), str(row["type"]), int(row["key"]))):
        key = int(item["key"])
        frame = int(item["frame"])
        if key < 0 or key >= 88 or frame < 0 or frame >= int(dense_total):
            continue
        start = max(frame - max(int(pre_dense_frames), 0), 0)
        end = min(frame + max(int(post_dense_frames), 0) + 1, int(dense_total))
        if end <= start:
            continue
        dedupe = (str(item["type"]), key, frame, start, end)
        if dedupe in seen:
            continue
        seen.add(dedupe)
        cleaned.append({**item, "start": int(start), "end": int(end)})
        if int(max_events) > 0 and len(cleaned) >= int(max_events):
            break
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Barcarolle: event-window ensemble repair for Impromptu hand states.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--candidate-npz", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pre-dense-frames", type=int, default=2)
    parser.add_argument("--post-dense-frames", type=int, default=10)
    parser.add_argument("--event-types", default="mispress,missed")
    parser.add_argument("--scopes", default="target_hand,full")
    parser.add_argument("--split-key", type=int, default=48)
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--max-total-evals", type=int, default=180)
    parser.add_argument("--min-delta-event-f1", type=float, default=1e-6)
    parser.add_argument("--allow-equal-f1-mispress-drop", action="store_true")
    parser.add_argument("--keep-if-matched-increases", action="store_true")
    parser.add_argument("--matched-gain-max-f1-drop", type=float, default=0.002)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source_npz)
    source = load_npz(source_path)
    dense = np.asarray(source["planned_hand_joints_dense"], dtype=np.float32).copy()
    target_keys = np.asarray(source["target_keys"], dtype=np.float32)[:, :88]
    control_steps = int(target_keys.shape[0])
    substeps = max(int(dense.shape[0] // max(control_steps, 1)), 1)
    dense_dt = 0.05 / float(substeps)

    candidates: list[tuple[str, np.ndarray]] = []
    for raw in args.candidate_npz:
        name, path = parse_candidate(str(raw))
        payload = load_npz(path)
        candidate_dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32)
        if candidate_dense.shape != dense.shape:
            raise ValueError(f"candidate {name} shape {candidate_dense.shape} != source shape {dense.shape}")
        if np.allclose(candidate_dense, dense, atol=1e-7, rtol=0.0):
            continue
        candidates.append((name, candidate_dense))
    if bool(args.include_neutral):
        neutral = np.repeat(dense[0].reshape(1, -1), dense.shape[0], axis=0).astype(np.float32)
        candidates.append(("neutral", neutral))
    if not candidates:
        raise ValueError("at least one non-identical candidate or --include-neutral is required")

    current, detail = simulate_score(source, dense, out / "eval_000_baseline", str(args.environment_name), float(args.threshold), dense_dt)
    initial = dict(current)
    event_types = {item.strip() for item in str(args.event_types).split(",") if item.strip()}
    scopes = [item.strip() for item in str(args.scopes).split(",") if item.strip()]
    windows = build_event_windows(
        detail,
        event_types=event_types,
        pre_dense_frames=int(args.pre_dense_frames),
        post_dense_frames=int(args.post_dense_frames),
        dense_total=int(dense.shape[0]),
        max_events=int(args.max_events),
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    selected = np.full((dense.shape[0],), -1, dtype=np.int32)
    eval_count = 0
    max_total = int(args.max_total_evals)

    for event_index, event in enumerate(windows):
        if max_total > 0 and eval_count >= max_total:
            break
        start = int(event["start"])
        end = int(event["end"])
        key = int(event["key"])
        previous = dense[start:end].copy()
        best_score: dict[str, Any] | None = None
        best_slice: np.ndarray | None = None
        best_name: str | None = None
        best_scope: str | None = None
        best_candidate_index = -1
        trials: list[dict[str, Any]] = []
        for candidate_index, (candidate_name, candidate_dense) in enumerate(candidates):
            for scope in scopes:
                if max_total > 0 and eval_count >= max_total:
                    break
                indices = indices_for_scope(str(scope), key, int(args.split_key))
                dense[start:end, indices] = candidate_dense[start:end, indices]
                eval_count += 1
                trial, _ = simulate_score(
                    source,
                    dense,
                    out / f"eval_{eval_count:04d}",
                    str(args.environment_name),
                    float(args.threshold),
                    dense_dt,
                )
                trials.append({"candidate": candidate_name, "scope": str(scope), "trial": trial})
                if best_score is None or score_key(trial) > score_key(best_score):
                    best_score = dict(trial)
                    best_slice = dense[start:end].copy()
                    best_name = str(candidate_name)
                    best_scope = str(scope)
                    best_candidate_index = int(candidate_index)
                dense[start:end] = previous
            if max_total > 0 and eval_count >= max_total:
                break
        delta = float(best_score["event_f1"] - current["event_f1"]) if best_score else 0.0
        accept = bool(best_score) and delta >= float(args.min_delta_event_f1)
        if (
            bool(best_score)
            and bool(args.allow_equal_f1_mispress_drop)
            and delta >= -1e-12
            and int(best_score["mispresses"]) < int(current["mispresses"])
        ):
            accept = True
        if (
            bool(best_score)
            and bool(args.keep_if_matched_increases)
            and int(best_score["matched"]) > int(current["matched"])
            and delta >= -abs(float(args.matched_gain_max_f1_drop))
        ):
            accept = True
        record = {
            "event_index": int(event_index),
            "event": {k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in event.items()},
            "previous": dict(current),
            "best_candidate": best_name,
            "best_scope": best_scope,
            "best_trial": best_score,
            "delta_event_f1": float(delta),
            "trials": trials,
        }
        if accept and best_slice is not None and best_score is not None:
            dense[start:end] = best_slice
            selected[start:end] = best_candidate_index
            current = best_score
            accepted.append(record)
        else:
            dense[start:end] = previous
            rejected.append(record)

    payload = dict(source)
    payload["planned_hand_joints_dense"] = dense.astype(np.float32)
    payload["planned_hand_velocities_dense"] = compute_hand_velocities(dense, control_timestep=float(dense_dt))
    payload["planned_hand_joints"] = dense[::substeps][:control_steps].astype(np.float32)
    payload["planned_hand_velocities"] = compute_hand_velocities(payload["planned_hand_joints"], control_timestep=0.05)
    payload["barcarolle_selected_candidate"] = selected
    atomic_save_npz(out / "trajectory.npz", **payload)
    final_score, _ = simulate_score(source, dense, out / "rp1m_sim", str(args.environment_name), float(args.threshold), dense_dt)
    result = {
        "source_npz": str(source_path),
        "candidate_npz": [str(item) for item in args.candidate_npz],
        "output_dir": str(out),
        "event_types": sorted(event_types),
        "scopes": scopes,
        "pre_dense_frames": int(args.pre_dense_frames),
        "post_dense_frames": int(args.post_dense_frames),
        "initial": initial,
        "greedy_score": current,
        "final_score": final_score,
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "event_window_count": int(len(windows)),
        "eval_count": int(eval_count),
        "accepted": accepted[:400],
        "rejected": rejected[:400],
    }
    atomic_save_json(out / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
