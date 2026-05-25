#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J fanhyagg
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/fanhyagg_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/fanhyagg_%j.err
#SBATCH -t 00:10:00
#SBATCH --cpus-per-task=1

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_fandango_hybrid_20260524

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
runs = []
for index in range(1, 11):
    run = f"maestro_{index:02d}"
    path = root / run / "hybrid" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    score = data["final_score"]
    row = {
        "run": run,
        "event_f1": float(score["event_f1"]),
        "frame_f1": float(score["frame_f1"]),
        "matched": int(score["matched"]),
        "target": int(score["target"]),
        "played": int(score["played"]),
        "mispresses": int(score["mispresses"]),
        "siciliana_accepts": int(json.loads((root / run / "siciliana_greedy" / "summary.json").read_text())["accepted_count"]),
        "cadenza_accepts": int(data["accepted_count"]),
    }
    runs.append(row)

event = np.asarray([row["event_f1"] for row in runs], dtype=np.float64)
frame = np.asarray([row["frame_f1"] for row in runs], dtype=np.float64)
summary = {
    "root": str(root),
    "runs": runs,
    "runs_completed": int(len(runs)),
    "mean_event_f1": float(event.mean()) if event.size else 0.0,
    "median_event_f1": float(np.median(event)) if event.size else 0.0,
    "mean_frame_f1": float(frame.mean()) if frame.size else 0.0,
    "median_frame_f1": float(np.median(frame)) if frame.size else 0.0,
    "total_matched": int(sum(row["matched"] for row in runs)),
    "total_target": int(sum(row["target"] for row in runs)),
    "total_played": int(sum(row["played"] for row in runs)),
    "total_mispresses": int(sum(row["mispresses"] for row in runs)),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
