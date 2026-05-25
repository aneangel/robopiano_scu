#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J bestsel30agg
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/bestsel30agg_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/bestsel30agg_%j.err
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/impromptu_best_bidirectional_barcarolle_rhapsody_30_20260524

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
missing = []
for idx in range(1, 31):
    run = f"maestro30_{idx:02d}"
    path = root / run / "selector_summary.json"
    if not path.exists():
        missing.append(run)
        continue
    row = json.loads(path.read_text())
    rows.append(row)

if len(rows) != 30:
    raise SystemExit(f"missing selector summaries: {missing}")

out = {
    "root": str(root),
    "model": "Impromptu bidirectional Barcarolle selector plus Rhapsody low-target repair",
    "run_count": len(rows),
    "mean_event_f1": sum(float(r["event_f1"]) for r in rows) / len(rows),
    "mean_frame_f1": sum(float(r["frame_f1"]) for r in rows) / len(rows),
    "matched": sum(int(r["matched"]) for r in rows),
    "target": sum(int(r["target"]) for r in rows),
    "played": sum(int(r["played"]) for r in rows),
    "mispresses": sum(int(r["mispresses"]) for r in rows),
    "runs": rows,
}
(root / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
