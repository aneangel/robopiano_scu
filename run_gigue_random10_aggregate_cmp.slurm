#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J gigue10agg
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/gigue10agg_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/gigue10agg_%j.err
#SBATCH -t 00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_gigue_chunk_20260524")
source_root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_fandango_hybrid_20260524")
rows = []
for run_index in range(1, 11):
    run = f"maestro_{run_index:02d}"
    path = root / run / "summary.json"
    if not path.exists():
        rows.append({"run": run, "missing": True})
        continue
    summary = json.loads(path.read_text())
    final_score = dict(summary["final_score"])
    initial = dict(summary["initial"])
    row = {
        "run": run,
        **final_score,
        "initial_event_f1": float(initial["event_f1"]),
        "initial_frame_f1": float(initial["frame_f1"]),
        "initial_matched": int(initial["matched"]),
        "initial_mispresses": int(initial["mispresses"]),
        "initial_played": int(initial["played"]),
        "accepted_count": int(summary.get("accepted_count", 0)),
        "rejected_count": int(summary.get("rejected_count", 0)),
    }
    row["delta_event_f1"] = float(row["event_f1"] - row["initial_event_f1"])
    row["delta_mispresses"] = int(row["mispresses"] - row["initial_mispresses"])
    rows.append(row)

valid = [row for row in rows if not row.get("missing")]
event_values = [float(row["event_f1"]) for row in valid]
frame_values = [float(row["frame_f1"]) for row in valid]
summary = {
    "root": str(root),
    "source_root": str(source_root),
    "runs_completed": len(valid),
    "runs": rows,
    "mean_event_f1": sum(event_values) / len(event_values) if event_values else 0.0,
    "mean_frame_f1": sum(frame_values) / len(frame_values) if frame_values else 0.0,
    "total_matched": sum(int(row["matched"]) for row in valid),
    "total_target": sum(int(row["target"]) for row in valid),
    "total_played": sum(int(row["played"]) for row in valid),
    "total_mispresses": sum(int(row["mispresses"]) for row in valid),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
