#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class SimpleCase:
    name: str
    controller: str
    kp: float
    kd: float
    ki: float
    setpoint_policy: str
    use_target_velocity: bool
    target_velocity_scale: float


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cases(args: argparse.Namespace) -> list[SimpleCase]:
    requested = {part.strip().lower() for part in str(args.controllers).split(",") if part.strip()}
    cases = [
        SimpleCase(
            name="p_linear_kp1",
            controller="p",
            kp=float(args.p_kp),
            kd=0.0,
            ki=0.0,
            setpoint_policy="linear",
            use_target_velocity=False,
            target_velocity_scale=0.0,
        ),
        SimpleCase(
            name="pd_linear_kp1_kd0005",
            controller="pd",
            kp=float(args.pd_kp),
            kd=float(args.pd_kd),
            ki=0.0,
            setpoint_policy="linear",
            use_target_velocity=False,
            target_velocity_scale=0.0,
        ),
    ]
    return [case for case in cases if case.controller in requested]


def _row_from_result(
    *,
    case: SimpleCase,
    trajectory: dict[str, Any],
    output_dir: Path,
    result: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    hand_l2 = (result or {}).get("hand_qpos_l2_vs_reference") or {}
    if not isinstance(hand_l2, dict):
        hand_l2 = {}
    return {
        "case": case.name,
        "controller": case.controller,
        "kp": case.kp,
        "kd": case.kd,
        "ki": case.ki,
        "setpoint_policy": case.setpoint_policy,
        "use_target_velocity": case.use_target_velocity,
        "target_velocity_scale": case.target_velocity_scale,
        "trajectory_npz": trajectory.get("trajectory_npz", ""),
        "environment_name": trajectory.get("environment_name", ""),
        "output_dir": str(output_dir),
        "result_json": str(output_dir / "pid_rollout_result.json"),
        "error": error or "",
        "source_steps_played": int((result or {}).get("source_steps_played") or 0),
        "actions_executed": int((result or {}).get("actions_executed") or 0),
        "hand_l2_mean": float(hand_l2.get("mean") or 0.0),
        "hand_l2_median": float(hand_l2.get("median") or 0.0),
        "hand_l2_max": float(hand_l2.get("max") or 0.0),
        "rp1m_key_f1": float((result or {}).get("rp1m_key_f1") or 0.0),
        "frame_f1": float((result or {}).get("frame_f1") or 0.0),
        "event_f1": float((result or {}).get("event_f1") or 0.0),
        "terminated": bool((result or {}).get("terminated")),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case",
        "controller",
        "kp",
        "kd",
        "setpoint_policy",
        "trajectory_npz",
        "environment_name",
        "source_steps_played",
        "actions_executed",
        "hand_l2_mean",
        "hand_l2_median",
        "hand_l2_max",
        "rp1m_key_f1",
        "frame_f1",
        "event_f1",
        "terminated",
        "error",
        "result_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simple P/PD sample-20 tests on prepared PID trajectories."
    )
    parser.add_argument("--trajectory-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--controllers", default="p,pd")
    parser.add_argument("--max-source-steps", type=int, default=20)
    parser.add_argument("--p-kp", type=float, default=1.0)
    parser.add_argument("--pd-kp", type=float, default=1.0)
    parser.add_argument("--pd-kd", type=float, default=0.005)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand-anchor-y-offset", type=float, default=None)
    parser.add_argument("--disable-hand-collisions", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = _read_json(Path(args.trajectory_manifest))
    trajectories = list(manifest.get("trajectories") or [])[: max(int(args.limit), 0)]
    if not trajectories:
        raise RuntimeError("Trajectory manifest contains no trajectories to test.")
    cases = _cases(args)
    if not cases:
        raise RuntimeError("No simple controller cases selected.")

    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []
    for trajectory_index, trajectory in enumerate(trajectories, start=1):
        trajectory_npz = Path(str(trajectory["trajectory_npz"]))
        env_name = str(
            trajectory.get("environment_name")
            or "RoboPianist-debug-NocturneRousseau-v0"
        )
        for case in cases:
            run_dir = output_root / f"t{trajectory_index:02d}_{case.name}"
            result_path = run_dir / "pid_rollout_result.json"
            if args.resume and result_path.exists():
                result = _read_json(result_path)
                error = None
            else:
                try:
                    result = run_impromptu_pid_rollout(
                        trajectory_npz=trajectory_npz,
                        output_dir=run_dir,
                        environment_name=env_name,
                        controller_kind=case.controller,  # type: ignore[arg-type]
                        kp=case.kp,
                        kd=case.kd,
                        ki=case.ki,
                        setpoint_policy=case.setpoint_policy,
                        use_target_velocity=case.use_target_velocity,
                        target_velocity_scale=case.target_velocity_scale,
                        threshold=float(args.threshold),
                        seed=int(args.seed),
                        max_source_steps=int(args.max_source_steps),
                        render_mp4=False,
                        hand_anchor_y_offset=args.hand_anchor_y_offset,
                        disable_hand_collisions=bool(args.disable_hand_collisions),
                    )
                    error = None
                except Exception as exc:
                    result = None
                    error = f"{type(exc).__name__}: {exc}"
            row = _row_from_result(
                case=case,
                trajectory=trajectory,
                output_dir=run_dir,
                result=result,
                error=error,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    event = np.asarray(
        [float(row["event_f1"]) for row in rows if not row["error"]],
        dtype=np.float64,
    )
    hand = np.asarray(
        [float(row["hand_l2_mean"]) for row in rows if not row["error"]],
        dtype=np.float64,
    )
    summary = {
        "trajectory_manifest": str(args.trajectory_manifest),
        "rows": rows,
        "successful_runs": int(sum(1 for row in rows if not row["error"])),
        "failed_runs": int(sum(1 for row in rows if row["error"])),
        "mean_event_f1": float(event.mean()) if event.size else 0.0,
        "mean_hand_l2": float(hand.mean()) if hand.size else 0.0,
    }
    save_json(output_root / "simple_test_summary.json", summary)
    _write_csv(output_root / "simple_test_summary.csv", rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
