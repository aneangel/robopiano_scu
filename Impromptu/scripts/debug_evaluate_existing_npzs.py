#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import strftime

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT,
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from impromptu.evaluation import evaluate_trajectory_npz  # noqa: E402


SEARCH_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu")
OUT_ROOT = SEARCH_ROOT / "test_runs"
METRIC_KEYS = (
    "assignment_rate",
    "waypoint_success_rate_010m",
    "waypoint_fingertip_error_p95",
    "exact_waypoint_sparse_error_p95",
    "exact_waypoint_anchor_error_p95",
    "exact_waypoint_anchor_success_rate_020m",
    "ik_anchor_error_weight_ge_1_p95",
    "ik_anchor_success_rate_020m_weight_ge_1",
    "ik_anchor_fingertip_distance_p95",
    "joint_velocity_p95",
    "joint_acceleration_p95",
    "joint_jerk_p95",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a small batch of existing Impromptu trajectory NPZ files.")
    parser.add_argument("--limit", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = sorted(SEARCH_ROOT.rglob("trajectory.npz"))[: max(int(args.limit), 0)]
    results: list[dict[str, object]] = []
    for path in paths:
        try:
            metrics = evaluate_trajectory_npz(path)
            results.append({"path": str(path), "ok": True, "metrics": {key: metrics.get(key) for key in METRIC_KEYS}})
            print(f"evaluated {path}")
        except Exception as exc:
            results.append({"path": str(path), "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"failed {path}: {type(exc).__name__}: {exc}")

    aggregate: dict[str, float] = {}
    successful = [item for item in results if item.get("ok")]
    for key in METRIC_KEYS:
        values = [
            float(item["metrics"][key])
            for item in successful
            if isinstance(item.get("metrics"), dict) and item["metrics"].get(key) is not None
        ]
        aggregate[f"{key}_mean"] = float(np.mean(values)) if values else 0.0

    for key in METRIC_KEYS:
        print(f"{key}_mean={aggregate[f'{key}_mean']:.6f}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"existing_npz_eval_{strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "search_root": str(SEARCH_ROOT),
                "limit": int(args.limit),
                "num_found": len(paths),
                "num_successful": len(successful),
                "aggregate": aggregate,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote existing NPZ evaluation summary: {out_path}")


if __name__ == "__main__":
    main()
