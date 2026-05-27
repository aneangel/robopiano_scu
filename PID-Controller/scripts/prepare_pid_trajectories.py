#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_SRC = Path(__file__).resolve().parents[1] / "src"
for path in (MODULE_SRC, REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from pid_controller.rollout import (  # noqa: E402
    DEFAULT_IMPROMPTU_RUN_ROOTS,
    discover_trajectory_npzs,
    rollout_rank_score,
    save_json,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _trajectory_info(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if "planned_hand_joints" in keys:
            hand_shape = list(np.asarray(data["planned_hand_joints"]).shape)
            hand_source = "planned_hand_joints"
            source_steps_20hz = int(hand_shape[0]) if len(hand_shape) >= 1 else 0
        elif "planned_hand_joints_dense" in keys:
            hand_shape = list(np.asarray(data["planned_hand_joints_dense"]).shape)
            hand_source = "planned_hand_joints_dense_downsampled_by_10"
            source_steps_20hz = int(hand_shape[0] // 10) if len(hand_shape) >= 1 else 0
        else:
            raise ValueError("missing planned_hand_joints and planned_hand_joints_dense")
        if len(hand_shape) != 2 or int(hand_shape[1]) != 46:
            raise ValueError(f"expected hand targets [T, 46], got {hand_shape}")
        target_shape = (
            list(np.asarray(data["target_keys"]).shape)
            if "target_keys" in keys
            else None
        )
    metadata = _read_json(path.parent / "metadata.json")
    return {
        "trajectory_npz": str(path),
        "run_dir": str(path.parent),
        "metadata_json": str(path.parent / "metadata.json"),
        "environment_name": metadata.get("environment_name"),
        "rank_score": float(rollout_rank_score(path)),
        "hand_source": hand_source,
        "hand_shape": hand_shape,
        "target_keys_shape": target_shape,
        "source_steps_20hz": source_steps_20hz,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "trajectory_npz",
        "environment_name",
        "rank_score",
        "hand_source",
        "hand_shape",
        "target_keys_shape",
        "source_steps_20hz",
        "metadata_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field))
                    if isinstance(row.get(field), (list, dict))
                    else row.get(field, "")
                    for field in fields
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a small manifest of planned trajectories for PID action tests."
    )
    parser.add_argument("--run-root", action="append", default=[])
    parser.add_argument("--trajectory-npz", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--environment-name", default="RoboPianist-debug-NocturneRousseau-v0")
    args = parser.parse_args()

    explicit = [Path(value) for value in args.trajectory_npz]
    roots = [Path(value) for value in args.run_root] or list(DEFAULT_IMPROMPTU_RUN_ROOTS)
    candidates = explicit or discover_trajectory_npzs(roots)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in candidates:
        if len(rows) >= max(int(args.limit), 0):
            break
        try:
            info = _trajectory_info(path)
            if not info.get("environment_name"):
                info["environment_name"] = str(args.environment_name)
            info["rank"] = len(rows) + 1
            rows.append(info)
        except Exception as exc:
            errors.append(
                {
                    "trajectory_npz": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not rows:
        raise RuntimeError(f"No valid planned trajectory.npz files prepared. Errors: {errors[:5]}")

    output_json = Path(args.output_json)
    payload = {
        "trajectories": rows,
        "errors": errors,
        "roots": [str(root) for root in roots],
        "limit": int(args.limit),
    }
    save_json(output_json, payload)
    csv_path = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    _write_csv(csv_path, rows)
    print(json.dumps({"manifest": str(output_json), "csv": str(csv_path), "count": len(rows)}))


if __name__ == "__main__":
    main()
