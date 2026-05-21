#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from submit_dense_sweep import write_sbatch_launcher
from sweep_common import DEFAULT_CODE_ROOT, DEFAULT_SWEEP_ROOT, json_dump, stage_output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or submit an adaptive dense sweep stage.")
    parser.add_argument("--base-root", default=str(DEFAULT_SWEEP_ROOT))
    parser.add_argument("--stage-name", default="stage2_adaptive")
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--partition", default="cmp")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--proposals", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1524)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_root).expanduser().resolve()
    output_root = stage_output_root(base_root, str(args.stage_name))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "slurm_logs").mkdir(parents=True, exist_ok=True)

    generator = [
        sys.executable,
        str(Path(args.code_root).expanduser().resolve() / "Impromptu/scripts/generate_adaptive_sweep_configs.py"),
        "--base-root",
        str(base_root),
        "--stage-name",
        str(args.stage_name),
        "--code-root",
        str(Path(args.code_root).expanduser().resolve()),
        "--top-k",
        str(int(args.top_k)),
        "--proposals",
        str(int(args.proposals)),
        "--seed",
        str(int(args.seed)),
    ]
    subprocess.run(generator, check=True)
    manifest = json.loads((output_root / "adaptive_manifest.json").read_text(encoding="utf-8"))
    max_index = int(manifest["config_count"]) - 1
    sbatch_path = output_root / "launch_dense_sweep_array.slurm"
    write_sbatch_launcher(
        sbatch_path=sbatch_path,
        partition=str(args.partition),
        max_index=max_index,
        output_root=output_root,
        code_root=Path(args.code_root).expanduser().resolve(),
    )
    summary = {"manifest": manifest, "sbatch_path": str(sbatch_path), "submitted": False}
    if args.submit:
        result = subprocess.run(["sbatch", str(sbatch_path)], check=True, capture_output=True, text=True)
        summary["submitted"] = True
        summary["sbatch_stdout"] = result.stdout.strip()
    json_dump(output_root / "submission_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
