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

from intermezzo.io import atomic_save_json  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from rp1m_simulator import RolloutConfig, make_rp1m_trajectory_from_arrays, simulate_rp1m_rollout  # noqa: E402


def _event_f1(score: dict[str, Any]) -> float:
    matched = float(score.get("matched_press_events", 0.0))
    played = float(score.get("played_press_events", 0.0))
    target = float(score.get("target_press_events", 0.0))
    precision = matched / played if played else 0.0
    recall = matched / target if target else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _dense_impromptu_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray, int, float]:
    with np.load(npz_path, allow_pickle=False) as data:
        hand_dense = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)
        target_keys = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
        control_steps = int(target_keys.shape[0])
        if control_steps <= 0:
            raise ValueError(f"{npz_path} has no target key frames")
        if hand_dense.shape[0] % control_steps != 0:
            raise ValueError(
                f"Cannot infer substeps: dense hand states {hand_dense.shape} vs target_keys {target_keys.shape}"
            )
        substeps = max(int(hand_dense.shape[0] // control_steps), 1)
        goals_dense = np.repeat(target_keys, substeps, axis=0).astype(np.float32)
    return hand_dense, goals_dense, substeps, 0.05 / float(substeps)


def run_one(
    *,
    trajectory_npz: Path,
    output_dir: Path,
    environment_name: str,
    threshold: float,
    seed: int,
    render_mp4: bool,
    max_source_steps: int | None,
    hand_anchor_y_offset: float | None,
    set_hand_qvel: bool,
) -> dict[str, Any]:
    hand_dense, goals_dense, substeps, dense_dt = _dense_impromptu_arrays(trajectory_npz)
    if max_source_steps is not None:
        keep = min(int(max_source_steps), int(hand_dense.shape[0]), int(goals_dense.shape[0]))
        hand_dense = hand_dense[:keep]
        goals_dense = goals_dense[:keep]
    actions = np.zeros((hand_dense.shape[0], 39), dtype=np.float32)
    trajectory = make_rp1m_trajectory_from_arrays(
        song_key=environment_name,
        demo_id=0,
        actions=actions,
        goals=goals_dense,
        hand_joints=hand_dense,
        environment_name=environment_name,
    )
    config = RolloutConfig(
        mode="hand_state",
        dataset_timestep=float(dense_dt),
        simulation_timestep=float(dense_dt),
        hand_anchor_y_offset=hand_anchor_y_offset,
        auto_hand_anchor_y_offset=False,
        hand_state_action_source="zero",
        restore_initial_hand=True,
        set_hand_qvel=bool(set_hand_qvel),
        seed=int(seed),
        threshold=float(threshold),
        max_source_steps=None,
        render_mp4=bool(render_mp4),
        render_audio=False,
        fps=int(round(1.0 / dense_dt)),
    )
    summary = simulate_rp1m_rollout(trajectory, config, output_dir)
    rollout_npz = Path(str(summary["rollout_npz"]))
    with np.load(rollout_npz, allow_pickle=False) as rollout:
        played = np.asarray(rollout["source_played_piano"], dtype=np.float32)
        goals = np.asarray(rollout["goals"], dtype=np.float32)
    score = score_rollout(
        target_keys=goals,
        played_keys=played,
        dt=float(dense_dt),
        threshold=float(threshold),
        timing_tolerance_s=0.15,
    )
    score_path = output_dir / "intermezzo_score.json"
    atomic_save_json(score_path, score)
    result = {
        "source_trajectory_npz": str(trajectory_npz),
        "run_dir": str(output_dir),
        "rp1m_summary_json": str(output_dir / "summary.json"),
        "intermezzo_score_json": str(score_path),
        "environment_name": environment_name,
        "dense_dt": float(dense_dt),
        "substeps": int(substeps),
        "source_steps_played": int(summary.get("source_steps_played", 0)),
        "rp1m_key_f1": float((summary.get("against_goals") or {}).get("key_f1", 0.0)),
        "rp1m_key_precision": float((summary.get("against_goals") or {}).get("key_precision", 0.0)),
        "rp1m_key_recall": float((summary.get("against_goals") or {}).get("key_recall", 0.0)),
        "frame_f1": float(score.get("frame_f1", 0.0)),
        "frame_precision": float(score.get("frame_precision", 0.0)),
        "frame_recall": float(score.get("frame_recall", 0.0)),
        "event_f1": float(_event_f1(score)),
        "matched": int(score.get("matched_press_events", 0)),
        "target": int(score.get("target_press_events", 0)),
        "missed": int(score.get("missed_key_presses", 0)),
        "mispresses": int(score.get("mispresses", 0)),
        "played": int(score.get("played_press_events", 0)),
        "terminated": bool(summary.get("terminated", False)),
        "hand_anchor_y_offset": hand_anchor_y_offset,
        "set_hand_qvel": bool(set_hand_qvel),
    }
    atomic_save_json(output_dir / "impromptu_rp1m_retest_result.json", result)
    return result


def _run_dirs(root: Path, only_run: str | None) -> list[Path]:
    if only_run:
        return [root / only_run]
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("maestro_"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retest Impromptu trajectories through rp1m_simulator.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--only-run", default=None)
    parser.add_argument("--environment-name", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-mp4", action="store_true")
    parser.add_argument("--max-source-steps", type=int, default=None)
    parser.add_argument("--use-rp1m-anchor-offset", action="store_true")
    parser.add_argument("--set-hand-qvel", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in _run_dirs(run_root, args.only_run):
        trajectory_npz = run_dir / "trajectory.npz"
        if not trajectory_npz.exists():
            continue
        metadata = _load_metadata(run_dir)
        env_name = str(args.environment_name or metadata.get("environment_name") or "RoboPianist-debug-NocturneRousseau-v0")
        out = output_root / run_dir.name
        print(f"=== retest {run_dir.name} env={env_name} ===", flush=True)
        row = run_one(
            trajectory_npz=trajectory_npz,
            output_dir=out,
            environment_name=env_name,
            threshold=float(args.threshold),
            seed=int(args.seed),
            render_mp4=bool(args.render_mp4),
            max_source_steps=args.max_source_steps,
            hand_anchor_y_offset=(-0.0490857549 if args.use_rp1m_anchor_offset else None),
            set_hand_qvel=bool(args.set_hand_qvel),
        )
        row["run"] = run_dir.name
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    frame = np.asarray([float(row["frame_f1"]) for row in rows], dtype=np.float64)
    event = np.asarray([float(row["event_f1"]) for row in rows], dtype=np.float64)
    rp1m_key = np.asarray([float(row["rp1m_key_f1"]) for row in rows], dtype=np.float64)
    summary = {
        "source_run_root": str(run_root),
        "output_root": str(output_root),
        "runs": rows,
        "mean_frame_f1": float(frame.mean()) if frame.size else 0.0,
        "median_frame_f1": float(np.median(frame)) if frame.size else 0.0,
        "mean_event_f1": float(event.mean()) if event.size else 0.0,
        "median_event_f1": float(np.median(event)) if event.size else 0.0,
        "mean_rp1m_key_f1": float(rp1m_key.mean()) if rp1m_key.size else 0.0,
        "median_rp1m_key_f1": float(np.median(rp1m_key)) if rp1m_key.size else 0.0,
        "total_matched": int(sum(int(row["matched"]) for row in rows)),
        "total_target": int(sum(int(row["target"]) for row in rows)),
        "total_mispresses": int(sum(int(row["mispresses"]) for row in rows)),
        "total_played": int(sum(int(row["played"]) for row in rows)),
    }
    atomic_save_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
