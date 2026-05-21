from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "run_name",
    "init_checkpoint",
    "refined_checkpoint",
    "controller_family",
    "weights_changed",
    "primary_metric",
    "piano/event_f1",
    "piano/missed_events",
    "piano/false_events",
    "piano/timing_abs_error_mean_s",
    "tracking/joint_pos_rmse",
    "fingertip/active_l2_mean",
    "action/l2_mean",
    "output_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Impromptu trajectory refinement runs.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output = Path(args.output)
    rows = collect_rows(runs_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_md = output.with_suffix(".md")
    best_json = output.parent / "best_overall.json"
    summary_md.write_text(render_markdown(rows, runs_root), encoding="utf-8")
    best_json.write_text(json.dumps(select_best(rows), indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output}")
    print(summary_md)
    print(best_json)


def collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not runs_root.exists():
        return rows
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = _read_yaml_like(run_dir / "config_resolved.yaml")
        refined = run_dir / "checkpoints" / "best_refined.pt"
        row = {
            "run_name": run_dir.name,
            "init_checkpoint": _extract_init_checkpoint(run_dir / "summary.md"),
            "refined_checkpoint": str(refined) if refined.exists() else "",
            "controller_family": _controller_family(config),
            "weights_changed": _bool_metric(metrics.get("refinement/weights_changed")),
            "primary_metric": _primary_metric(metrics),
            "output_path": str(run_dir),
        }
        for key in FIELDS:
            if key in metrics:
                row[key] = metrics[key]
        rows.append({key: row.get(key, "") for key in FIELDS})
    return rows


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def as_float(row: dict[str, Any], key: str, default: float) -> float:
        try:
            if row.get(key) == "":
                return default
            return float(row.get(key))
        except (TypeError, ValueError):
            return default

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        event_f1 = as_float(row, "piano/event_f1", float("-inf"))
        joint_rmse = as_float(row, "tracking/joint_pos_rmse", float("inf"))
        missed = as_float(row, "piano/missed_events", float("inf"))
        false = as_float(row, "piano/false_events", float("inf"))
        return (event_f1, -joint_rmse, -missed, -false)

    return max(rows, key=key)


def render_markdown(rows: list[dict[str, Any]], runs_root: Path) -> str:
    lines = [
        "# Impromptu Refinement Summary",
        "",
        f"- Runs root: `{runs_root}`",
        f"- Completed runs found: `{len(rows)}`",
        "",
    ]
    if not rows:
        lines.append("No completed refinement runs with `metrics.json` were found.")
        return "\n".join(lines) + "\n"
    lines.append("| run | weights changed | event_f1 | joint_rmse | checkpoint |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            "| {run} | {changed} | {f1} | {rmse} | `{ckpt}` |".format(
                run=row["run_name"],
                changed=row["weights_changed"],
                f1=row["piano/event_f1"],
                rmse=row["tracking/joint_pos_rmse"],
                ckpt=row["refined_checkpoint"],
            )
        )
    best = select_best(rows)
    if best:
        lines.extend(["", f"Best overall: `{best['run_name']}`"])
    return "\n".join(lines) + "\n"


def _extract_init_checkpoint(summary_path: Path) -> str:
    if not summary_path.exists():
        return ""
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        prefix = "- Init checkpoint: `"
        if line.startswith(prefix) and line.endswith("`"):
            return line[len(prefix) : -1]
    return ""


def _primary_metric(metrics: dict[str, Any]) -> str:
    if "piano/event_f1" in metrics:
        return "piano/event_f1"
    if "tracking/joint_pos_rmse" in metrics:
        return "tracking/joint_pos_rmse"
    if "fingertip/active_l2_mean" in metrics:
        return "fingertip/active_l2_mean"
    return ""


def _bool_metric(value: Any) -> str:
    try:
        return "true" if float(value) > 0.5 else "false"
    except (TypeError, ValueError):
        return ""


def _controller_family(config: dict[str, Any]) -> str:
    controller = config.get("controller", {}) if isinstance(config, dict) else {}
    if not isinstance(controller, dict):
        return ""
    return str(controller.get("family") or controller.get("type") or "")


def _read_yaml_like(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


if __name__ == "__main__":
    main()
