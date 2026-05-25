#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J best30agg
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/best30_agg_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/best30_agg_%j.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python

"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random30_best_adaptive_20260524")
manifest = json.loads((root / "manifest.json").read_text())
rows = []
missing = []
for index in range(1, 31):
    run = f"maestro30_{index:02d}"
    path = root / run / "best_summary.json"
    if not path.exists():
        missing.append(run)
        continue
    rows.append(json.loads(path.read_text()))
event = np.asarray([float(row["event_f1"]) for row in rows], dtype=np.float64)
frame = np.asarray([float(row["frame_f1"]) for row in rows], dtype=np.float64)
summary = {
    "root": str(root),
    "manifest": manifest,
    "runs": rows,
    "runs_completed": int(len(rows)),
    "missing_runs": missing,
    "mean_event_f1": float(event.mean()) if event.size else 0.0,
    "median_event_f1": float(np.median(event)) if event.size else 0.0,
    "std_event_f1": float(event.std(ddof=0)) if event.size else 0.0,
    "mean_frame_f1": float(frame.mean()) if frame.size else 0.0,
    "median_frame_f1": float(np.median(frame)) if frame.size else 0.0,
    "total_matched": int(sum(int(row["matched"]) for row in rows)),
    "total_target": int(sum(int(row["target"]) for row in rows)),
    "total_played": int(sum(int(row["played"]) for row in rows)),
    "total_mispresses": int(sum(int(row["mispresses"]) for row in rows)),
    "choice_counts": {
        "legato_caprice": int(sum(1 for row in rows if row.get("choice") == "legato_caprice")),
        "rondo": int(sum(1 for row in rows if row.get("choice") == "rondo")),
    },
    "comparison_10_song_best_mean_event_f1": 0.6500640620142223,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if missing:
    raise SystemExit(f"Missing 30-song summaries: {missing}")
PY
