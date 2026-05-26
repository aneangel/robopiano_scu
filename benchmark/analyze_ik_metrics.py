"""Summarize per-waypoint IK metrics produced by the smoke planner run.

Reads the ``ik_metrics.npy`` array written by
``Bagatelle/src/bagatelle/planner.py`` (column layout documented in the
``IK_METRIC_COLUMNS`` constant of that module) and prints aggregate
statistics: per-column mean / median / p95 / p99 / max, plus a few
derived quantities (total nfev, mean nfev, success fraction). The same
summary is written to ``benchmark/results/ik_metrics_summary.txt``.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np

METRICS_PATH = "/tmp/maestroso_smoke/twinkle_smoke/ik_metrics.npy"
OUTPUT_PATH = "/Users/aangeles/robopiano/benchmark/results/ik_metrics_summary.txt"

# Mirrors Bagatelle/src/bagatelle/planner.py::IK_METRIC_COLUMNS.
IK_METRIC_COLUMNS: Sequence[str] = (
    "success",
    "optimizer_success",
    "nfev",
    "optimizer_cost",
    "mean_assigned_distance",
    "max_assigned_distance",
    "residual_norm",
)


def _column_stats(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"  {name}: <empty>"
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return f"  {name}: all non-finite"
    return (
        f"  {name}: "
        f"mean={float(np.mean(finite)):.6g} "
        f"median={float(np.median(finite)):.6g} "
        f"p95={float(np.percentile(finite, 95)):.6g} "
        f"p99={float(np.percentile(finite, 99)):.6g} "
        f"max={float(np.max(finite)):.6g} "
        f"n={finite.size}"
    )


def main() -> int:
    if not os.path.exists(METRICS_PATH):
        print(f"[analyze_ik_metrics] missing {METRICS_PATH}", file=sys.stderr)
        return 1

    arr = np.load(METRICS_PATH)
    if arr.ndim != 2 or arr.shape[1] != len(IK_METRIC_COLUMNS):
        print(
            f"[analyze_ik_metrics] unexpected shape {arr.shape}; "
            f"expected (N, {len(IK_METRIC_COLUMNS)})",
            file=sys.stderr,
        )
        return 1

    lines: list[str] = []
    lines.append(f"ik_metrics_path: {METRICS_PATH}")
    lines.append(f"num_waypoints: {arr.shape[0]}")
    lines.append("per-column statistics:")
    for idx, name in enumerate(IK_METRIC_COLUMNS):
        lines.append(_column_stats(name, arr[:, idx]))

    success = arr[:, IK_METRIC_COLUMNS.index("success")]
    nfev = arr[:, IK_METRIC_COLUMNS.index("nfev")]
    total_nfev = float(np.sum(nfev))
    mean_nfev = float(np.mean(nfev)) if nfev.size else float("nan")
    if success.size:
        success_fraction = float(np.mean(success >= 0.5))
    else:
        success_fraction = float("nan")

    lines.append("")
    lines.append("derived:")
    lines.append(f"  total_nfev: {total_nfev:.0f}")
    lines.append(f"  mean_nfev_per_waypoint: {mean_nfev:.4f}")
    lines.append(f"  success_fraction: {success_fraction:.4f}")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
