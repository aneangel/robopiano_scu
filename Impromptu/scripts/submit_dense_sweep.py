#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from sweep_common import DEFAULT_CODE_ROOT, DEFAULT_SWEEP_ROOT, json_dump


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=imp_dense_sweep
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-{max_index}
#SBATCH --output={output_root}/slurm_logs/%A_%a.out
#SBATCH --error={output_root}/slurm_logs/%A_%a.err

set -euxo pipefail

cd {code_root}
eval "$(conda shell.bash hook)"
conda activate sonata

export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export MUJOCO_GL=egl
export LD_PRELOAD="${{CONDA_PREFIX}}/lib/libstdc++.so.6${{LD_PRELOAD:+:${{LD_PRELOAD}}}}"

CONFIG=$(printf "{output_root}/configs/%03d.json" "${{SLURM_ARRAY_TASK_ID}}")
python Impromptu/scripts/run_dense_sweep_job.py --config "${{CONFIG}}"
"""


def write_sbatch_launcher(*, sbatch_path: Path, partition: str, max_index: int, output_root: Path, code_root: Path) -> None:
    sbatch_path.write_text(
        SBATCH_TEMPLATE.format(
            partition=str(partition),
            max_index=int(max_index),
            output_root=str(output_root),
            code_root=str(code_root),
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sweep configs and submit an sbatch array.")
    parser.add_argument("--output-root", default=str(DEFAULT_SWEEP_ROOT))
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--partition", default="cmp")
    parser.add_argument("--random-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=524)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "slurm_logs").mkdir(parents=True, exist_ok=True)
    generator = [
        sys.executable,
        str(Path(args.code_root) / "Impromptu/scripts/generate_dense_sweep_configs.py"),
        "--output-root",
        str(output_root),
        "--code-root",
        str(Path(args.code_root).expanduser().resolve()),
        "--random-count",
        str(int(args.random_count)),
        "--seed",
        str(int(args.seed)),
    ]
    subprocess.run(generator, check=True)
    manifest = json.loads((output_root / "sweep_manifest.json").read_text(encoding="utf-8"))
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
