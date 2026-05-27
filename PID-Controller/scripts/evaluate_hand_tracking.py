#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_SRC = Path(__file__).resolve().parents[1] / "src"
for path in (
    MODULE_SRC,
    REPO_ROOT,
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "partita" / "src",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from pid_controller.rollout import run_impromptu_pid_rollout, save_json  # noqa: E402
from pid_controller.mapping import build_reduced_action_mapping, joint_index_groups  # noqa: E402


SOURCE_DT = 0.05
SIM_DT = 0.005
SUBSTEPS = int(round(SOURCE_DT / SIM_DT))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_source_and_dense_targets(
    trajectory_npz: Path,
    *,
    source_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(trajectory_npz, allow_pickle=False) as data:
        if "planned_hand_joints" in data:
            hand_20hz = np.asarray(data["planned_hand_joints"], dtype=np.float32)
        elif "planned_hand_joints_dense" in data:
            hand_20hz = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)[::SUBSTEPS]
        else:
            raise ValueError(f"{trajectory_npz} has no planned hand-state targets")
        if "planned_hand_joints_dense" in data:
            hand_dense = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)
        else:
            hand_dense = np.repeat(hand_20hz, SUBSTEPS, axis=0)
        if "target_keys" not in data:
            raise ValueError(f"{trajectory_npz} has no target_keys")
        goals_20hz = np.asarray(data["target_keys"], dtype=np.float32)[:, :88]
    steps = min(int(source_steps), int(hand_20hz.shape[0]), int(goals_20hz.shape[0]))
    if steps < 2:
        raise ValueError(f"Need at least two 20Hz source steps, got {steps}")
    dense_steps = min(steps * SUBSTEPS, int(hand_dense.shape[0]))
    dense_source_steps = dense_steps // SUBSTEPS
    steps = min(steps, dense_source_steps)
    dense_steps = steps * SUBSTEPS
    return (
        hand_20hz[:steps].astype(np.float32),
        goals_20hz[:steps].astype(np.float32),
        hand_dense[:dense_steps].astype(np.float32),
        np.repeat(goals_20hz[:steps], SUBSTEPS, axis=0).astype(np.float32),
    )


def _make_rp1m_trajectory(
    *,
    environment_name: str,
    hand: np.ndarray,
    goals: np.ndarray,
    action_dim: int,
):
    from rp1m_simulator import make_rp1m_trajectory_from_arrays

    return make_rp1m_trajectory_from_arrays(
        song_key=str(environment_name),
        demo_id=0,
        actions=np.zeros((hand.shape[0], action_dim), dtype=np.float32),
        goals=goals.astype(np.float32),
        hand_joints=hand.astype(np.float32),
        environment_name=str(environment_name),
    )


def _run_hand_state_rollout(
    *,
    trajectory_npz: Path,
    output_dir: Path,
    environment_name: str,
    source_steps: int,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    from rp1m_simulator import RolloutConfig, simulate_rp1m_rollout

    _hand_20hz, _goals_20hz, hand_dense, goals_dense = _load_source_and_dense_targets(
        trajectory_npz,
        source_steps=source_steps,
    )
    trajectory = _make_rp1m_trajectory(
        environment_name=environment_name,
        hand=hand_dense,
        goals=goals_dense,
        action_dim=39,
    )
    config = RolloutConfig(
        mode="hand_state",
        dataset_timestep=SIM_DT,
        simulation_timestep=SIM_DT,
        hand_anchor_y_offset=None,
        auto_hand_anchor_y_offset=False,
        action_source_scale="actuator_units",
        hand_state_action_source="zero",
        restore_initial_hand=True,
        set_hand_qvel=True,
        seed=int(seed),
        threshold=float(threshold),
        max_source_steps=None,
        render_mp4=False,
        render_audio=False,
    )
    summary = simulate_rp1m_rollout(trajectory, config, output_dir)
    summary["source_20hz_steps_requested"] = int(source_steps)
    summary["trajectory_npz"] = str(trajectory_npz)
    save_json(output_dir / "hand_state_result.json", summary)
    return summary


def _tracking_series(
    *,
    rollout_npz: Path,
    target_hand_20hz: np.ndarray,
    target_goals_20hz: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(rollout_npz, allow_pickle=False) as data:
        actual = np.asarray(data["source_hand_after_step"], dtype=np.float32)
        played = np.asarray(data["source_played_piano"], dtype=np.float32)
    reference = target_hand_20hz[1 : actual.shape[0] + 1]
    n = min(int(actual.shape[0]), int(reference.shape[0]))
    if n <= 0:
        return [], {
            "scored_steps": 0,
            "alignment": "after_action_t_vs_target_t_plus_1",
        }
    actual = actual[:n]
    reference = reference[:n]
    diff = actual - reference
    l2 = np.linalg.norm(diff, axis=1)
    groups = joint_index_groups(build_reduced_action_mapping())
    one_to_one_indices = list(groups["one_to_one"])
    coupled_indices = list(groups["coupled"])
    one_to_one_l2 = (
        np.linalg.norm(diff[:, one_to_one_indices], axis=1)
        if one_to_one_indices
        else np.zeros((n,), dtype=np.float32)
    )
    coupled_l2 = (
        np.linalg.norm(diff[:, coupled_indices], axis=1)
        if coupled_indices
        else np.zeros((n,), dtype=np.float32)
    )
    mean_abs = np.mean(np.abs(diff), axis=1)
    max_abs = np.max(np.abs(diff), axis=1)
    rows: list[dict[str, Any]] = []
    for index in range(n):
        source_step = index
        target_step = index + 1
        rows.append(
            {
                "source_step": int(source_step),
                "target_step": int(target_step),
                "time_s": float(target_step * SOURCE_DT),
                "l2": float(l2[index]),
                "one_to_one_l2": float(one_to_one_l2[index]),
                "coupled_l2": float(coupled_l2[index]),
                "mean_abs": float(mean_abs[index]),
                "max_abs": float(max_abs[index]),
                "target_active_keys": int((target_goals_20hz[target_step] > 0.5).sum())
                if target_step < target_goals_20hz.shape[0]
                else 0,
                "played_active_keys": int((played[source_step] > 0.5).sum())
                if source_step < played.shape[0]
                else 0,
                "cumulative_l2_mean": float(l2[: index + 1].mean()),
            }
        )
    times = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    slope = float(np.polyfit(times, l2.astype(np.float64), deg=1)[0]) if n >= 2 else 0.0
    split = max(n // 3, 1)
    first = l2[:split]
    last = l2[-split:]
    summary = {
        "alignment": "after_action_t_vs_target_t_plus_1",
        "scored_steps": int(n),
        "one_to_one_joint_count": int(len(one_to_one_indices)),
        "coupled_joint_count": int(len(coupled_indices)),
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
        "l2_max": float(l2.max()),
        "l2_first_third_mean": float(first.mean()),
        "l2_last_third_mean": float(last.mean()),
        "l2_last_minus_first": float(last.mean() - first.mean()),
        "l2_last_over_first": float(last.mean() / max(float(first.mean()), 1e-8)),
        "l2_slope_per_second": slope,
        "peak_l2_step": int(rows[int(np.argmax(l2))]["target_step"]),
        "final_l2": float(l2[-1]),
        "final_cumulative_l2_mean": float(l2.mean()),
        "one_to_one_l2_mean": float(one_to_one_l2.mean()),
        "one_to_one_l2_median": float(np.median(one_to_one_l2)),
        "one_to_one_l2_max": float(one_to_one_l2.max()),
        "one_to_one_final_l2": float(one_to_one_l2[-1]),
        "coupled_l2_mean": float(coupled_l2.mean()),
        "coupled_l2_median": float(np.median(coupled_l2)),
        "coupled_l2_max": float(coupled_l2.max()),
        "coupled_final_l2": float(coupled_l2[-1]),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _summarize_rollout(summary: dict[str, Any]) -> dict[str, Any]:
    goals = summary.get("against_goals") or {}
    hand_l2 = summary.get("hand_qpos_l2_vs_reference") or {}
    return {
        "source_steps_played": int(summary.get("source_steps_played") or 0),
        "dense_steps_played": int(summary.get("dense_steps_played") or 0),
        "actions_executed": int(summary.get("actions_executed") or 0),
        "terminated": bool(summary.get("terminated")),
        "key_f1": float(goals.get("key_f1") or 0.0),
        "key_precision": float(goals.get("key_precision") or 0.0),
        "key_recall": float(goals.get("key_recall") or 0.0),
        "hand_l2": hand_l2,
        "rollout_npz": summary.get("rollout_npz"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hand-state validity and action-control tracking drift."
    )
    parser.add_argument("--trajectory-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-source-steps", type=int, default=20)
    parser.add_argument("--controller", choices=["p", "pd", "pid"], default="pd")
    parser.add_argument("--kp", type=float, default=1.3)
    parser.add_argument("--kd", type=float, default=0.005)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--setpoint-policy", choices=["next", "linear", "minimum_jerk"], default="minimum_jerk")
    parser.add_argument("--target-velocity-scale", type=float, default=0.0)
    parser.add_argument("--use-target-velocity", action="store_true")
    parser.add_argument("--feedforward-scale", type=float, default=0.0)
    parser.add_argument("--lookahead-substeps", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    manifest = _read_json(Path(args.trajectory_manifest))
    records = list(manifest.get("trajectories") or [])[: max(int(args.limit), 0)]
    if not records:
        raise RuntimeError("No trajectories in manifest.")
    output_root = Path(args.output_root)
    aggregate_rows: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        trajectory_npz = Path(str(record["trajectory_npz"]))
        environment_name = str(record.get("environment_name") or "RoboPianist-debug-NocturneRousseau-v0")
        run_name = trajectory_npz.parent.name
        run_root = output_root / f"{index:02d}_{run_name}"
        source_steps = min(int(args.max_source_steps), int(record.get("source_steps_20hz") or args.max_source_steps))
        hand_20hz, goals_20hz, _hand_dense, _goals_dense = _load_source_and_dense_targets(
            trajectory_npz,
            source_steps=source_steps,
        )

        hand_state_dir = run_root / "hand_state_dense"
        hand_state_summary_path = hand_state_dir / "summary.json"
        if args.resume and hand_state_summary_path.exists():
            hand_state_summary = _read_json(hand_state_summary_path)
        else:
            hand_state_summary = _run_hand_state_rollout(
                trajectory_npz=trajectory_npz,
                output_dir=hand_state_dir,
                environment_name=environment_name,
                source_steps=source_steps,
                threshold=float(args.threshold),
                seed=int(args.seed),
            )

        action_dir = (
            run_root
            / f"action_{args.controller}_kp{args.kp:g}_kd{args.kd:g}_"
            f"tv{args.target_velocity_scale:g}_ff{args.feedforward_scale:g}_h{args.lookahead_substeps:g}"
        )
        action_result_path = action_dir / "pid_rollout_result.json"
        if args.resume and action_result_path.exists():
            action_result = _read_json(action_result_path)
        else:
            action_result = run_impromptu_pid_rollout(
                trajectory_npz=trajectory_npz,
                output_dir=action_dir,
                environment_name=environment_name,
                controller_kind=args.controller,
                kp=float(args.kp),
                kd=float(args.kd),
                ki=float(args.ki),
                setpoint_policy=args.setpoint_policy,
                use_target_velocity=bool(args.use_target_velocity),
                target_velocity_scale=float(args.target_velocity_scale),
                feedforward_scale=float(args.feedforward_scale),
                lookahead_substeps=int(args.lookahead_substeps),
                threshold=float(args.threshold),
                seed=int(args.seed),
                max_source_steps=source_steps,
                render_mp4=False,
                hand_anchor_y_offset=None,
            )

        tracking_rows, tracking_summary = _tracking_series(
            rollout_npz=Path(str(action_result["rollout_npz"])),
            target_hand_20hz=hand_20hz,
            target_goals_20hz=goals_20hz,
        )
        _write_csv(run_root / "action_tracking_by_20hz_step.csv", tracking_rows)
        save_json(run_root / "action_tracking_summary.json", tracking_summary)

        row = {
            "rank": int(index),
            "trajectory_npz": str(trajectory_npz),
            "run_name": run_name,
            "source_steps": int(source_steps),
            "hand_state_key_f1": _summarize_rollout(hand_state_summary)["key_f1"],
            "hand_state_terminated": _summarize_rollout(hand_state_summary)["terminated"],
            "hand_state_summary_json": str(hand_state_summary_path),
            "action_event_f1": float(action_result.get("event_f1") or 0.0),
            "action_frame_f1": float(action_result.get("frame_f1") or 0.0),
            "action_key_f1": float(action_result.get("rp1m_key_f1") or 0.0),
            "action_terminated": bool(action_result.get("terminated")),
            "action_result_json": str(action_result_path),
            **tracking_summary,
        }
        aggregate_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    _write_csv(output_root / "tracking_aggregate.csv", aggregate_rows)
    save_json(
        output_root / "tracking_aggregate.json",
        {
            "rows": aggregate_rows,
            "mean_action_key_f1": float(np.mean([row["action_key_f1"] for row in aggregate_rows])),
            "mean_l2": float(np.mean([row["l2_mean"] for row in aggregate_rows])),
            "mean_l2_slope_per_second": float(
                np.mean([row["l2_slope_per_second"] for row in aggregate_rows])
            ),
        },
    )


if __name__ == "__main__":
    main()
