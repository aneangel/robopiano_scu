#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "Bagatelle" / "src", REPO_ROOT / "Variations" / "src", REPO_ROOT):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

try:
    from variations.utils.io import ensure_dir, save_json
except Exception:
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _float_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean_m": float("nan"),
            "median_m": float("nan"),
            "p90_m": float("nan"),
            "p95_m": float("nan"),
            "p99_m": float("nan"),
            "max_m": float("nan"),
            "success_at_0p005": float("nan"),
            "success_at_0p01": float("nan"),
            "success_at_0p02": float("nan"),
            "success_at_0p05": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "p95_m": float(np.percentile(values, 95)),
        "p99_m": float(np.percentile(values, 99)),
        "max_m": float(np.max(values)),
        "success_at_0p005": float(np.mean(values <= 0.005)),
        "success_at_0p01": float(np.mean(values <= 0.010)),
        "success_at_0p02": float(np.mean(values <= 0.020)),
        "success_at_0p05": float(np.mean(values <= 0.050)),
    }


def collect_distances(trajectory_npz: str | Path, *, label: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    path = Path(trajectory_npz).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        targets = np.asarray(data["fingertip_targets"], dtype=np.float32)
        measured = np.asarray(data["waypoint_fingertips"], dtype=np.float32)
        assignments = np.asarray(data["assignments"], dtype=np.int32) if "assignments" in data.files else None
        waypoint_frames = np.asarray(data["waypoint_frames"], dtype=np.int32) if "waypoint_frames" in data.files else None
    if targets.shape != measured.shape or targets.ndim != 3 or targets.shape[1:] != (10, 3):
        raise ValueError(f"{path} must contain fingertip_targets and waypoint_fingertips with shape [W, 10, 3]")
    mask = np.isfinite(targets).all(axis=2)
    diff = measured - targets
    center = np.linalg.norm(diff, axis=2)
    xy = np.linalg.norm(diff[:, :, :2], axis=2)
    z = np.abs(diff[:, :, 2])
    # In RoboPianist/Bagatelle key pitch runs along world y; x is front/back.
    width = np.abs(diff[:, :, 1])
    front = np.abs(diff[:, :, 0])
    rows: list[dict[str, Any]] = []
    for waypoint_index, finger_index in np.argwhere(mask):
        key_index = int(assignments[waypoint_index, finger_index]) if assignments is not None else -1
        frame = int(waypoint_frames[waypoint_index]) if waypoint_frames is not None else int(waypoint_index)
        rows.append(
            {
                "label": label,
                "trajectory_npz": str(path),
                "waypoint_index": int(waypoint_index),
                "frame": frame,
                "finger_index": int(finger_index),
                "key_index": key_index,
                "center_distance_m": float(center[waypoint_index, finger_index]),
                "surface_distance_m": float(center[waypoint_index, finger_index]),
                "xy_distance_m": float(xy[waypoint_index, finger_index]),
                "z_distance_m": float(z[waypoint_index, finger_index]),
                "front_distance_m": float(front[waypoint_index, finger_index]),
                "width_center_distance_m": float(width[waypoint_index, finger_index]),
                "width_surface_distance_m": float(width[waypoint_index, finger_index]),
            }
        )
    return rows, {
        "center": center[mask].astype(np.float64),
        "surface": center[mask].astype(np.float64),
        "xy": xy[mask].astype(np.float64),
        "z": z[mask].astype(np.float64),
        "width_center": width[mask].astype(np.float64),
        "width_surface": width[mask].astype(np.float64),
        "front": front[mask].astype(np.float64),
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "trajectory_npz",
        "waypoint_index",
        "frame",
        "finger_index",
        "key_index",
        "center_distance_m",
        "surface_distance_m",
        "xy_distance_m",
        "z_distance_m",
        "front_distance_m",
        "width_center_distance_m",
        "width_surface_distance_m",
    ]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_histograms(path: Path, series: dict[str, dict[str, np.ndarray]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    bins = np.linspace(0.0, 0.30, 61)
    colors = {"baseline": "#808080", "lookahead": "#2f6f9f", "sequence": "#2f6f9f"}
    for ax, metric in zip(axes, ("center", "surface"), strict=True):
        for label, values_by_metric in series.items():
            values = values_by_metric[metric]
            if values.size == 0:
                continue
            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.48,
                color=colors.get(label, None),
                edgecolor="black",
                linewidth=0.25,
                label=f"{label} (n={values.size})",
            )
        ax.axvline(0.005, color="#666666", linestyle="--", linewidth=1.0)
        ax.axvline(0.010, color="#666666", linestyle=":", linewidth=1.0)
        ax.axvline(0.020, color="#666666", linestyle="-.", linewidth=1.0)
        ax.set_title(f"{metric.capitalize()} distance")
        ax.set_xlabel("Selected fingertip distance to target key (m)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Bagatelle selected fingertip-to-target-key distance histograms.")
    parser.add_argument("--trajectory", action="append", required=True, help="LABEL=trajectory.npz. Can be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="bagatelle_fingertip_distance_histogram")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    all_rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, np.ndarray]] = {}
    trajectories: dict[str, str] = {}
    for spec in args.trajectory:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = Path(path).parent.name or Path(path).stem
        rows, distances = collect_distances(path, label=label)
        all_rows.extend(rows)
        series[label] = distances
        trajectories[label] = str(Path(path).expanduser().resolve())

    summary = {
        "trajectories": trajectories,
        "series": {label: {metric: _float_summary(values) for metric, values in values_by_metric.items()} for label, values_by_metric in series.items()},
    }
    png_path = output_dir / f"{args.prefix}.png"
    csv_path = output_dir / f"{args.prefix}.csv"
    json_path = output_dir / f"{args.prefix}.json"
    plot_histograms(png_path, series, title="Bagatelle selected fingertip distances")
    write_rows_csv(csv_path, all_rows)
    save_json(json_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved histogram: {png_path}")
    print(f"Saved distances CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    main()
