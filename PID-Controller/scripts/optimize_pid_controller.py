#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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

from pid_controller.optimization import (  # noqa: E402
    GainCandidate,
    make_gain_candidates,
    metric_from_row,
    score_pid_result,
)
from pid_controller.rollout import (  # noqa: E402
    DEFAULT_IMPROMPTU_RUN_ROOTS,
    discover_trajectory_npzs,
    rollout_rank_score,
    run_impromptu_pid_rollout,
    save_json,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _float_grid(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in _split_csv(value))


def _environment_for(path: Path, fallback: str) -> str:
    metadata = _read_json(path.parent / "metadata.json")
    return str(metadata.get("environment_name") or fallback)


def _select_trajectories(args: argparse.Namespace) -> list[Path]:
    selected = [Path(value) for value in args.trajectory_npz]
    if selected:
        return selected[: max(int(args.trajectory_limit), 0)]
    roots = [Path(value) for value in args.run_root] or list(DEFAULT_IMPROMPTU_RUN_ROOTS)
    return discover_trajectory_npzs(roots)[: max(int(args.trajectory_limit), 0)]


def _candidate_from_row(row: dict[str, Any]) -> GainCandidate:
    def as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    return GainCandidate(
        controller_kind=str(row["controller_kind"]),
        kp=float(row["kp"]),
        kd=float(row["kd"]),
        ki=float(row["ki"]),
        integral_limit=float(row["integral_limit"]),
        setpoint_policy=str(row["setpoint_policy"]),
        use_target_velocity=as_bool(row["use_target_velocity"]),
        target_velocity_scale=float(row["target_velocity_scale"]),
        feedforward_scale=float(row.get("feedforward_scale") or 0.0),
        lookahead_substeps=int(float(row.get("lookahead_substeps") or 10)),
        sustain_value=float(row["sustain_value"]),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    else:
        fields = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _result_row(
    *,
    candidate_index: int,
    candidate: GainCandidate,
    trajectory_npz: Path,
    output_dir: Path,
    result: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    hand_l2 = (result or {}).get("hand_qpos_l2_vs_reference") or {}
    if not isinstance(hand_l2, dict):
        hand_l2 = {}
    split = (result or {}).get("hand_tracking_split") or {}
    if not isinstance(split, dict):
        split = {}
    one_to_one = split.get("one_to_one_qpos_l2") or {}
    coupled = split.get("coupled_qpos_l2") or {}
    actuator = split.get("actuator_signal_l2") or {}
    total = split.get("total_qpos_l2") or {}
    if not isinstance(one_to_one, dict):
        one_to_one = {}
    if not isinstance(coupled, dict):
        coupled = {}
    if not isinstance(actuator, dict):
        actuator = {}
    if not isinstance(total, dict):
        total = {}
    row = {
        "candidate_index": int(candidate_index),
        **candidate.to_dict(),
        "trajectory_npz": str(trajectory_npz),
        "prior_rank_score": float(rollout_rank_score(trajectory_npz)),
        "output_dir": str(output_dir),
        "result_json": str(output_dir / "pid_rollout_result.json"),
        "error": error or "",
        "objective": 0.0,
        "hand_l2_mean": 0.0,
        "hand_l2_median": 0.0,
        "hand_l2_max": 0.0,
        "hand_l2_final": 0.0,
        "hand_l2_last_third_mean": 0.0,
        "hand_l2_last_minus_first": 0.0,
        "one_to_one_l2_mean": 0.0,
        "one_to_one_l2_median": 0.0,
        "one_to_one_l2_max": 0.0,
        "one_to_one_l2_final": 0.0,
        "one_to_one_l2_last_third_mean": 0.0,
        "coupled_l2_mean": 0.0,
        "coupled_l2_median": 0.0,
        "coupled_l2_max": 0.0,
        "coupled_l2_final": 0.0,
        "coupled_l2_last_third_mean": 0.0,
        "actuator_l2_mean": 0.0,
        "actuator_l2_median": 0.0,
        "actuator_l2_max": 0.0,
        "rp1m_key_f1": 0.0,
        "frame_f1": 0.0,
        "event_f1": 0.0,
        "source_steps_played": 0,
        "actions_executed": 0,
        "terminated": False,
    }
    if result is not None and error is None:
        row.update(
            {
                "objective": float(score_pid_result(result)),
                "hand_l2_mean": float(hand_l2.get("mean") or 0.0),
                "hand_l2_median": float(hand_l2.get("median") or 0.0),
                "hand_l2_max": float(hand_l2.get("max") or 0.0),
                "hand_l2_final": float(total.get("final") or 0.0),
                "hand_l2_last_third_mean": float(total.get("last_third_mean") or 0.0),
                "hand_l2_last_minus_first": float(total.get("last_minus_first") or 0.0),
                "one_to_one_l2_mean": float(one_to_one.get("mean") or 0.0),
                "one_to_one_l2_median": float(one_to_one.get("median") or 0.0),
                "one_to_one_l2_max": float(one_to_one.get("max") or 0.0),
                "one_to_one_l2_final": float(one_to_one.get("final") or 0.0),
                "one_to_one_l2_last_third_mean": float(one_to_one.get("last_third_mean") or 0.0),
                "coupled_l2_mean": float(coupled.get("mean") or 0.0),
                "coupled_l2_median": float(coupled.get("median") or 0.0),
                "coupled_l2_max": float(coupled.get("max") or 0.0),
                "coupled_l2_final": float(coupled.get("final") or 0.0),
                "coupled_l2_last_third_mean": float(coupled.get("last_third_mean") or 0.0),
                "actuator_l2_mean": float(actuator.get("mean") or 0.0),
                "actuator_l2_median": float(actuator.get("median") or 0.0),
                "actuator_l2_max": float(actuator.get("max") or 0.0),
                "rp1m_key_f1": float(result.get("rp1m_key_f1") or 0.0),
                "frame_f1": float(result.get("frame_f1") or 0.0),
                "event_f1": float(result.get("event_f1") or 0.0),
                "source_steps_played": int(result.get("source_steps_played") or 0),
                "actions_executed": int(result.get("actions_executed") or 0),
                "terminated": bool(result.get("terminated")),
            }
        )
    return row


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["candidate_index"])].append(row)
    aggregates: list[dict[str, Any]] = []
    metric_keys = (
        "objective",
        "hand_l2_mean",
        "hand_l2_median",
        "hand_l2_max",
        "hand_l2_final",
        "hand_l2_last_third_mean",
        "hand_l2_last_minus_first",
        "one_to_one_l2_mean",
        "one_to_one_l2_median",
        "one_to_one_l2_max",
        "one_to_one_l2_final",
        "one_to_one_l2_last_third_mean",
        "coupled_l2_mean",
        "coupled_l2_median",
        "coupled_l2_max",
        "coupled_l2_final",
        "coupled_l2_last_third_mean",
        "actuator_l2_mean",
        "actuator_l2_median",
        "actuator_l2_max",
        "rp1m_key_f1",
        "frame_f1",
        "event_f1",
    )
    for candidate_index, candidate_rows in sorted(grouped.items()):
        first = candidate_rows[0]
        ok_rows = [row for row in candidate_rows if not row.get("error")]
        aggregate = {
            "candidate_index": int(candidate_index),
            "controller_kind": first["controller_kind"],
            "kp": first["kp"],
            "kd": first["kd"],
            "ki": first["ki"],
            "integral_limit": first["integral_limit"],
            "setpoint_policy": first["setpoint_policy"],
            "use_target_velocity": first["use_target_velocity"],
            "target_velocity_scale": first["target_velocity_scale"],
            "feedforward_scale": first["feedforward_scale"],
            "lookahead_substeps": first["lookahead_substeps"],
            "sustain_value": first["sustain_value"],
            "runs": int(len(candidate_rows)),
            "successful_runs": int(len(ok_rows)),
            "failed_runs": int(len(candidate_rows) - len(ok_rows)),
            "terminated_runs": int(sum(1 for row in ok_rows if bool(row.get("terminated")))),
        }
        for key in metric_keys:
            values = np.asarray([float(row.get(key) or 0.0) for row in ok_rows], dtype=np.float64)
            aggregate[f"mean_{key}"] = float(values.mean()) if values.size else 0.0
            aggregate[f"median_{key}"] = float(np.median(values)) if values.size else 0.0
        aggregates.append(aggregate)
    return aggregates


def _run_candidate(
    *,
    candidate_index: int,
    candidate: GainCandidate,
    trajectory_npz: Path,
    output_dir: Path,
    environment_name: str,
    args: argparse.Namespace,
    full_song: bool,
) -> dict[str, Any]:
    if args.resume and (output_dir / "pid_rollout_result.json").exists():
        result = _read_json(output_dir / "pid_rollout_result.json")
        return _result_row(
            candidate_index=candidate_index,
            candidate=candidate,
            trajectory_npz=trajectory_npz,
            output_dir=output_dir,
            result=result,
            error=None,
        )
    try:
        result = run_impromptu_pid_rollout(
            trajectory_npz=trajectory_npz,
            output_dir=output_dir,
            environment_name=environment_name,
            controller_kind=candidate.controller_kind,  # type: ignore[arg-type]
            kp=candidate.kp,
            kd=candidate.kd,
            ki=candidate.ki,
            integral_limit=candidate.integral_limit,
            setpoint_policy=candidate.setpoint_policy,
            use_target_velocity=candidate.use_target_velocity,
            target_velocity_scale=candidate.target_velocity_scale,
            feedforward_scale=candidate.feedforward_scale,
            lookahead_substeps=candidate.lookahead_substeps,
            sustain_value=candidate.sustain_value,
            threshold=float(args.threshold),
            seed=int(args.seed),
            max_source_steps=None if full_song else int(args.max_source_steps),
            render_mp4=False,
            hand_anchor_y_offset=args.hand_anchor_y_offset,
            disable_hand_collisions=bool(args.disable_hand_collisions),
        )
        return _result_row(
            candidate_index=candidate_index,
            candidate=candidate,
            trajectory_npz=trajectory_npz,
            output_dir=output_dir,
            result=result,
            error=None,
        )
    except Exception as exc:
        return _result_row(
            candidate_index=candidate_index,
            candidate=candidate,
            trajectory_npz=trajectory_npz,
            output_dir=output_dir,
            result=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search PID gains against planned Impromptu/Maestroso trajectories."
    )
    parser.add_argument("--run-root", action="append", default=[])
    parser.add_argument("--trajectory-npz", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trajectory-limit", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--controllers", default="pd")
    parser.add_argument("--kp-grid", default="0.7,1.0,1.3,1.6")
    parser.add_argument("--kd-grid", default="0,0.005,0.01,0.02")
    parser.add_argument("--ki-grid", default="0.0,0.01,0.03")
    parser.add_argument("--integral-limit-grid", default="0.1")
    parser.add_argument("--setpoint-policies", default="minimum_jerk")
    parser.add_argument("--target-velocity-scales", default="0,0.5,1.0")
    parser.add_argument("--feedforward-scales", default="0.0")
    parser.add_argument("--lookahead-substeps-grid", default="10")
    parser.add_argument("--sustain-values", default="0.0")
    parser.add_argument(
        "--selection-metric",
        choices=[
            "objective",
            "event_f1",
            "frame_f1",
            "rp1m_key_f1",
            "hand_l2_mean",
            "hand_l2_final",
            "hand_l2_last_third_mean",
            "one_to_one_l2_mean",
            "one_to_one_l2_final",
            "one_to_one_l2_last_third_mean",
            "coupled_l2_mean",
            "actuator_l2_mean",
        ],
        default="objective",
    )
    parser.add_argument("--max-source-steps", type=int, default=20)
    parser.add_argument("--scale-best", action="store_true")
    parser.add_argument("--scale-limit", type=int, default=1)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand-anchor-y-offset", type=float, default=None)
    parser.add_argument("--disable-hand-collisions", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trajectories = _select_trajectories(args)
    if not trajectories:
        raise RuntimeError("No trajectory.npz files found for optimization.")
    candidates = make_gain_candidates(
        controllers=_split_csv(args.controllers),
        kp_values=_float_grid(args.kp_grid),
        kd_values=_float_grid(args.kd_grid),
        ki_values=_float_grid(args.ki_grid),
        integral_limits=_float_grid(args.integral_limit_grid),
        setpoint_policies=_split_csv(args.setpoint_policies),
        target_velocity_scales=_float_grid(args.target_velocity_scales),
        feedforward_scales=_float_grid(args.feedforward_scales),
        lookahead_substeps_values=tuple(int(float(value)) for value in _split_csv(args.lookahead_substeps_grid)),
        sustain_values=_float_grid(args.sustain_values),
    )
    if args.candidate_limit is not None:
        candidates = candidates[: max(int(args.candidate_limit), 0)]
    if not candidates:
        raise RuntimeError("No PID gain candidates selected.")

    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        for trajectory_index, trajectory_npz in enumerate(trajectories, start=1):
            env_name = str(args.environment_name)
            if args.environment_name == "RoboPianist-debug-NocturneRousseau-v0":
                env_name = _environment_for(trajectory_npz, args.environment_name)
            safe_traj = trajectory_npz.parent.name.replace("/", "_").replace("\\", "_")
            label = candidate.label(candidate_index)
            run_dir = output_root / "sample20" / f"t{trajectory_index:02d}_{safe_traj}" / label
            print(
                f"candidate={candidate_index}/{len(candidates)} trajectory={trajectory_index}/"
                f"{len(trajectories)} {label} {trajectory_npz}",
                flush=True,
            )
            row = _run_candidate(
                candidate_index=candidate_index,
                candidate=candidate,
                trajectory_npz=trajectory_npz,
                output_dir=run_dir,
                environment_name=env_name,
                args=args,
                full_song=False,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            _write_csv(output_root / "candidate_results.csv", rows)
            save_json(output_root / "candidate_results.json", {"rows": rows})

    aggregates = _aggregate_rows(rows)
    if not any(int(row.get("successful_runs") or 0) > 0 for row in aggregates):
        _write_csv(output_root / "candidate_aggregates.csv", aggregates)
        save_json(
            output_root / "optimization_summary.json",
            {
                "error": "all_candidate_rollouts_failed",
                "aggregates": aggregates,
                "candidate_count": int(len(candidates)),
                "trajectory_count": int(len(trajectories)),
                "max_source_steps": int(args.max_source_steps),
            },
        )
        raise RuntimeError("All PID candidate rollouts failed; inspect candidate_results.csv.")
    aggregates = sorted(
        aggregates,
        key=lambda row: (
            metric_from_row(row, args.selection_metric),
            float(row.get("mean_event_f1") or 0.0),
            float(row.get("mean_rp1m_key_f1") or 0.0),
            -float(row.get("mean_hand_l2_mean") or 0.0),
        ),
        reverse=True,
    )
    best_row = aggregates[0]
    best_candidate = _candidate_from_row(best_row)
    summary: dict[str, Any] = {
        "selection_metric": args.selection_metric,
        "best_candidate": best_candidate.to_dict(),
        "best_aggregate": best_row,
        "aggregates": aggregates,
        "candidate_count": int(len(candidates)),
        "trajectory_count": int(len(trajectories)),
        "max_source_steps": int(args.max_source_steps),
    }
    _write_csv(output_root / "candidate_aggregates.csv", aggregates)
    save_json(output_root / "optimization_summary.json", summary)
    print(
        json.dumps(
            {"best": summary["best_candidate"], "aggregate": best_row},
            indent=2,
            sort_keys=True,
        )
    )

    if args.scale_best:
        scale_trajectories = trajectories[: max(int(args.scale_limit), 0)]
        scale_rows: list[dict[str, Any]] = []
        for trajectory_index, trajectory_npz in enumerate(scale_trajectories, start=1):
            env_name = str(args.environment_name)
            if args.environment_name == "RoboPianist-debug-NocturneRousseau-v0":
                env_name = _environment_for(trajectory_npz, args.environment_name)
            safe_traj = trajectory_npz.parent.name.replace("/", "_").replace("\\", "_")
            run_dir = output_root / "scale_best_full_song" / f"t{trajectory_index:02d}_{safe_traj}"
            row = _run_candidate(
                candidate_index=int(best_row["candidate_index"]),
                candidate=best_candidate,
                trajectory_npz=trajectory_npz,
                output_dir=run_dir,
                environment_name=env_name,
                args=args,
                full_song=True,
            )
            scale_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        scale_aggregates = _aggregate_rows(scale_rows)
        summary["scale_best"] = {
            "rows": scale_rows,
            "aggregates": scale_aggregates,
        }
        _write_csv(output_root / "scale_best_results.csv", scale_rows)
        save_json(output_root / "optimization_summary.json", summary)


if __name__ == "__main__":
    main()
