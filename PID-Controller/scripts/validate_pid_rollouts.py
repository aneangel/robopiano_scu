#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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

from pid_controller.rollout import (  # noqa: E402
    DEFAULT_IMPROMPTU_RUN_ROOTS,
    discover_trajectory_npzs,
    rollout_rank_score,
    run_impromptu_pid_rollout,
    save_json,
)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _environment_for(path: Path, fallback: str) -> str:
    metadata = _read_json(path.parent / "metadata.json")
    return str(metadata.get("environment_name") or fallback)


def _manifest_records(path: str | Path, *, limit: int) -> list[dict[str, object]]:
    manifest = _read_json(Path(path))
    records = list(manifest.get("trajectories") or [])
    return records[: max(int(limit), 0)]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "trajectory_npz",
        "controller_kind",
        "source_steps_played",
        "actions_executed",
        "hand_l2_mean",
        "hand_l2_median",
        "hand_l2_max",
        "one_to_one_l2_mean",
        "coupled_l2_mean",
        "rp1m_key_f1",
        "frame_f1",
        "event_f1",
        "terminated",
        "result_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PID controller on the best Impromptu/Maestroso rollouts."
    )
    parser.add_argument("--run-root", action="append", default=[])
    parser.add_argument("--trajectory-npz", action="append", default=[])
    parser.add_argument("--trajectory-manifest", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--controller", choices=["p", "pd", "pid"], default="pd")
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--ki", type=float, default=None)
    parser.add_argument("--integral-limit", type=float, default=0.25)
    parser.add_argument("--setpoint-policy", choices=["next", "linear", "minimum_jerk"], default="minimum_jerk")
    parser.add_argument("--target-velocity-scale", type=float, default=0.0)
    parser.add_argument("--no-target-velocity", dest="use_target_velocity", action="store_false")
    parser.add_argument("--use-target-velocity", dest="use_target_velocity", action="store_true")
    parser.set_defaults(use_target_velocity=False)
    parser.add_argument("--feedforward-scale", type=float, default=0.0)
    parser.add_argument("--lookahead-substeps", type=int, default=10)
    parser.add_argument("--sustain-value", type=float, default=0.0)
    parser.add_argument("--max-source-steps", type=int, default=20)
    parser.add_argument("--full-song", action="store_true")
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand-anchor-y-offset", type=float, default=None)
    parser.add_argument("--disable-hand-collisions", action="store_true")
    args = parser.parse_args()

    roots = [Path(value) for value in args.run_root] or list(DEFAULT_IMPROMPTU_RUN_ROOTS)
    selected_records: list[dict[str, object]] = [
        {"trajectory_npz": str(value)}
        for value in args.trajectory_npz
    ]
    if not selected_records and args.trajectory_manifest:
        selected_records = _manifest_records(args.trajectory_manifest, limit=int(args.limit))
    if not selected_records:
        selected_records = [
            {"trajectory_npz": str(path)}
            for path in discover_trajectory_npzs(roots)[: max(int(args.limit), 0)]
        ]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for rank, record in enumerate(selected_records, start=1):
        trajectory_npz = Path(str(record["trajectory_npz"]))
        env_name = str(args.environment_name)
        if record.get("environment_name"):
            env_name = str(record["environment_name"])
        elif args.environment_name == "RoboPianist-debug-NocturneRousseau-v0":
            env_name = _environment_for(trajectory_npz, args.environment_name)
        safe_name = trajectory_npz.parent.name.replace("/", "_").replace("\\", "_")
        run_dir = output_root / f"{rank:02d}_{safe_name}_{args.controller}"
        print(
            f"=== rank={rank} score={rollout_rank_score(trajectory_npz):.4f} "
            f"controller={args.controller} trajectory={trajectory_npz} ===",
            flush=True,
        )
        result = run_impromptu_pid_rollout(
            trajectory_npz=trajectory_npz,
            output_dir=run_dir,
            environment_name=env_name,
            controller_kind=args.controller,
            kp=args.kp,
            kd=args.kd,
            ki=args.ki,
            integral_limit=float(args.integral_limit),
            setpoint_policy=args.setpoint_policy,
            use_target_velocity=bool(args.use_target_velocity),
            target_velocity_scale=float(args.target_velocity_scale),
            feedforward_scale=float(args.feedforward_scale),
            lookahead_substeps=int(args.lookahead_substeps),
            sustain_value=float(args.sustain_value),
            threshold=float(args.threshold),
            seed=int(args.seed),
            max_source_steps=None if args.full_song else int(args.max_source_steps),
            render_mp4=False,
            hand_anchor_y_offset=args.hand_anchor_y_offset,
            disable_hand_collisions=bool(args.disable_hand_collisions),
        )
        hand_l2 = result.get("hand_qpos_l2_vs_reference") or {}
        split = result.get("hand_tracking_split") or {}
        one_to_one = (split.get("one_to_one_qpos_l2") or {}) if isinstance(split, dict) else {}
        coupled = (split.get("coupled_qpos_l2") or {}) if isinstance(split, dict) else {}
        row = {
            "rank": int(rank),
            "trajectory_npz": str(trajectory_npz),
            "controller_kind": str(args.controller),
            "source_steps_played": int(result.get("source_steps_played", 0)),
            "actions_executed": int(result.get("actions_executed", 0)),
            "hand_l2_mean": float(hand_l2.get("mean", 0.0)),
            "hand_l2_median": float(hand_l2.get("median", 0.0)),
            "hand_l2_max": float(hand_l2.get("max", 0.0)),
            "one_to_one_l2_mean": float(one_to_one.get("mean", 0.0)),
            "coupled_l2_mean": float(coupled.get("mean", 0.0)),
            "rp1m_key_f1": float(result.get("rp1m_key_f1", 0.0)),
            "frame_f1": float(result.get("frame_f1", 0.0)),
            "event_f1": float(result.get("event_f1", 0.0)),
            "terminated": bool(result.get("terminated", False)),
            "result_json": str(run_dir / "pid_rollout_result.json"),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    event = np.asarray([float(row["event_f1"]) for row in rows], dtype=np.float64)
    hand = np.asarray([float(row["hand_l2_mean"]) for row in rows], dtype=np.float64)
    summary = {
        "controller": args.controller,
        "rows": rows,
        "mean_event_f1": float(event.mean()) if event.size else 0.0,
        "median_event_f1": float(np.median(event)) if event.size else 0.0,
        "mean_hand_l2": float(hand.mean()) if hand.size else 0.0,
        "median_hand_l2": float(np.median(hand)) if hand.size else 0.0,
    }
    save_json(output_root / "summary.json", summary)
    _write_csv(output_root / "summary.csv", rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
