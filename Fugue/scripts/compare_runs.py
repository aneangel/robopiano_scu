#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fugue.comparison import compare_runs, discover_run_dirs  # noqa: E402
from fugue.constants import DEFAULT_OUTPUT_ROOT, DEFAULT_RP1M_ROOT  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare completed Fugue training runs.")
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_OUTPUT_ROOT / "runs"),
        help="Root containing Fugue run directories.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Specific top-level or model-level run directory. Can be repeated.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rp1m-root", default=str(DEFAULT_RP1M_ROOT))
    parser.add_argument("--evaluate-missing-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--complete-only", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = [*args.run_dir] if args.run_dir else [args.runs_root]
    if args.list_only:
        for path in discover_run_dirs(inputs):
            print(path)
        return
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / "comparisons" / datetime.now().strftime("%Y%m%d_%H%M%S")
    result = compare_runs(
        run_dirs=inputs,
        output_dir=output_dir,
        dataset_root=args.rp1m_root,
        evaluate_missing_test=bool(args.evaluate_missing_test),
        batch_size=int(args.batch_size),
        device=args.device,
        include_incomplete=not bool(args.complete_only),
    )
    print(f"records={len(result['records'])}")
    print(f"csv={result['csv_path']}")
    print(f"json={result['json_path']}")
    print(f"report={result['report_path']}")
    print(result["recommendation"]["recommendation"])


if __name__ == "__main__":
    main()
