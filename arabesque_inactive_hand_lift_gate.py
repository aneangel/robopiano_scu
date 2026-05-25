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


RIGHT_FOREARM_TY_INDEX = 22
LEFT_FOREARM_TY_INDEX = 45


def event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0))
    played = float(score.get("played_press_events", 0))
    target = float(score.get("target_press_events", 0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def lift_control(
    control: np.ndarray,
    target_keys: np.ndarray,
    *,
    split_key: int,
    lift: float,
    mode: str,
    kin: BagatelleKinematics,
    threshold: float,
) -> tuple[np.ndarray, int]:
    out = np.asarray(control, dtype=np.float32).copy()
    changed = 0
    for frame, row in enumerate(np.asarray(target_keys, dtype=np.float32)[:, :88]):
        keys = np.flatnonzero(row > float(threshold))
        if mode == "both":
            lift_left = True
            lift_right = True
        elif keys.size == 0:
            lift_left = True
            lift_right = True
        else:
            has_left = bool(np.any(keys < int(split_key)))
            has_right = bool(np.any(keys >= int(split_key)))
            if mode == "inactive":
                lift_left = has_right and not has_left
                lift_right = has_left and not has_right
            elif mode == "left":
                lift_left = True
                lift_right = False
            elif mode == "right":
                lift_left = False
                lift_right = True
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        before = out[frame].copy()
        if lift_left:
            out[frame, LEFT_FOREARM_TY_INDEX] += np.float32(lift)
        if lift_right:
            out[frame, RIGHT_FOREARM_TY_INDEX] += np.float32(lift)
        out[frame] = kin.clip_qpos(out[frame])
        if not np.allclose(before, out[frame], atol=1e-7, rtol=0.0):
            changed += 1
    return out.astype(np.float32), int(changed)


def score_variant(
    *,
    payload: dict[str, np.ndarray],
    control: np.ndarray,
    target_keys: np.ndarray,
    substeps: int,
    out: Path,
    environment_name: str,
    threshold: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    dense = np.repeat(control, substeps, axis=0).astype(np.float32)
    dense_goals = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    dense_dt = 0.05 / float(substeps)
    variant_payload = {key: np.asarray(value) for key, value in payload.items()}
    variant_payload["planned_hand_joints"] = control.astype(np.float32)
    variant_payload["planned_hand_joints_dense"] = dense
    atomic_save_npz(out / "trajectory.npz", **variant_payload)
    traj = make_rp1m_trajectory_from_arrays(
        song_key=str(environment_name),
        demo_id=0,
        actions=np.zeros((dense.shape[0], 39), dtype=np.float32),
        goals=dense_goals,
        hand_joints=dense,
        environment_name=str(environment_name),
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
            threshold=float(threshold),
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
        threshold=float(threshold),
        timing_tolerance_s=0.15,
    )
    result = {
        **metadata,
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "event_f1": float(event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "rp1m_key_f1": float((sim_summary.get("against_goals") or {}).get("key_f1", 0.0)),
    }
    atomic_save_json(out / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Arabesque: lift inactive-hand forearm during one-hand passages.")
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-key", type=int, default=44)
    parser.add_argument("--lifts", type=float, nargs="+", default=[0.005, 0.010, 0.015])
    parser.add_argument("--modes", nargs="+", default=["inactive"])
    args = parser.parse_args()

    source = Path(args.source_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    control = np.asarray(payload["planned_hand_joints"], dtype=np.float32)
    target_keys = np.asarray(payload["target_keys"], dtype=np.float32)[:, :88]
    dense = np.asarray(payload["planned_hand_joints_dense"], dtype=np.float32)
    substeps = max(int(dense.shape[0] // max(control.shape[0], 1)), 1)

    rows = []
    config = BagatelleConfig(environment_name=str(args.environment_name), threshold=float(args.threshold), seed=0)
    with BagatelleKinematics(config=config, target_keys=target_keys, output_dir=out / "kinematics") as kin:
        for mode in args.modes:
            for lift in args.lifts:
                variant_control, changed = lift_control(
                    control,
                    target_keys,
                    split_key=int(args.split_key),
                    lift=float(lift),
                    mode=str(mode),
                    kin=kin,
                    threshold=float(args.threshold),
                )
                variant_dir = out / f"{mode}_lift_{float(lift):.4f}".replace(".", "p")
                variant_dir.mkdir(parents=True, exist_ok=True)
                row = score_variant(
                    payload=payload,
                    control=variant_control,
                    target_keys=target_keys,
                    substeps=substeps,
                    out=variant_dir,
                    environment_name=str(args.environment_name),
                    threshold=float(args.threshold),
                    metadata={
                        "source_npz": str(source),
                        "output_dir": str(variant_dir),
                        "mode": str(mode),
                        "lift": float(lift),
                        "changed_control_frames": int(changed),
                    },
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    best = max(rows, key=lambda row: (float(row["event_f1"]), float(row["frame_f1"]))) if rows else None
    summary = {"source_npz": str(source), "output_dir": str(out), "rows": rows, "best": best}
    atomic_save_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
