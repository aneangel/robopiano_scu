#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", REPO_ROOT / "Fugue" / "src", REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from allegro.hand_error import (  # noqa: E402
    DEFAULT_PREFIX_STEPS,
    compute_source_hand_errors,
    load_rollout_hand_arrays,
    summarize_error_prefixes,
    write_error_csv,
    write_summary_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize how far each simulated hand state drifts from the RP1M planner hand state."
    )
    parser.add_argument("--rollout-npz", action="append", required=True, help="Rollout NPZ with sim/target hand states.")
    parser.add_argument("--label", action="append", default=None, help="Optional label for each rollout NPZ.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--target-offset", type=int, default=1)
    parser.add_argument(
        "--prefix-steps",
        default=",".join(str(value) for value in DEFAULT_PREFIX_STEPS),
        help="Comma-separated source-step horizons for prefix summaries.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(value) for value in args.rollout_npz]
    labels = list(args.label or [])
    if labels and len(labels) != len(paths):
        raise ValueError("--label must be supplied once per --rollout-npz")
    if not labels:
        labels = [_default_label(path) for path in paths]
    prefixes = _parse_prefixes(args.prefix_steps)
    runs: list[dict[str, Any]] = []
    for label, path in zip(labels, paths):
        sim, target = load_rollout_hand_arrays(path)
        errors = compute_source_hand_errors(
            sim_hand_joints=sim,
            target_hand_joints=target,
            dt=float(args.dt),
            target_offset=int(args.target_offset),
        )
        csv_path = write_error_csv(output_dir / f"{label}_hand_error_by_step.csv", errors)
        prefix_rows = summarize_error_prefixes(errors, prefixes=prefixes)
        runs.append(
            {
                "label": label,
                "rollout_npz": str(path),
                "hand_error_csv": str(csv_path),
                "target_definition": (
                    "sim_hand_joints[i] is compared to target_hand_joints[i + target_offset]; "
                    "with target_offset=1 this measures error after executing source interval i "
                    "against the next RP1M planner endpoint"
                ),
                "target_offset": int(args.target_offset),
                "dt": float(args.dt),
                "source_steps": int(errors["hand_l2"].shape[0]),
                "prefix_summaries": prefix_rows,
            }
        )
    summary = {"runs": runs}
    summary_path = write_summary_json(output_dir / "hand_error_summary.json", summary)
    print(f"summary_path={summary_path}")
    for run in runs:
        last = run["prefix_summaries"][-1] if run["prefix_summaries"] else {}
        print(
            f"{run['label']} steps={run['source_steps']} "
            f"mean={last.get('hand_l2_mean')} p95={last.get('hand_l2_p95')} "
            f"final={last.get('hand_l2_final')}"
        )


def _default_label(path: Path) -> str:
    text = path.parent.name or path.stem
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


def _parse_prefixes(value: str) -> list[int]:
    prefixes = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        prefixes.append(int(item))
    if not prefixes:
        raise ValueError("At least one prefix step must be supplied")
    return prefixes


if __name__ == "__main__":
    main()
