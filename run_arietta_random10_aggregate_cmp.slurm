#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J arietta10agg
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/arietta10_agg_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/arietta10_agg_%j.err
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_arietta_margin_20260524")
base_path = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_adaptive_legato_caprice_rondo_20260524/summary.json")
base = json.loads(base_path.read_text())
base_rows = {row["run"]: row for row in base["runs"]}
rows = []
for index in range(1, 11):
    run = f"maestro_{index:02d}"
    path = root / run / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    final = data["final_score"]
    base_row = base_rows[run]
    row = {
        "run": run,
        "event_f1": float(final["event_f1"]),
        "frame_f1": float(final["frame_f1"]),
        "matched": int(final["matched"]),
        "target": int(final["target"]),
        "played": int(final["played"]),
        "mispresses": int(final["mispresses"]),
        "accepted_count": int(data.get("accepted_count", 0)),
        "eval_count": int(data.get("eval_count", 0)),
        "source_event_f1": float(base_row["event_f1"]),
        "source_frame_f1": float(base_row["frame_f1"]),
        "source_matched": int(base_row["matched"]),
        "source_mispresses": int(base_row["mispresses"]),
    }
    row["delta_event_f1"] = float(row["event_f1"] - row["source_event_f1"])
    row["delta_mispresses"] = int(row["mispresses"] - row["source_mispresses"])
    rows.append(row)

out = {
    "root": str(root),
    "source_summary": str(base_path),
    "policy": "Arietta lateral-margin sweep on the adaptive Legato-Caprice/Rondo source. For active-target mispress events, sweep small forearm ty offsets on the target hand and keep only simulator-scored event-F1 or equal-F1 precision improvements.",
    "runs": rows,
    "runs_completed": len(rows),
    "total_matched": int(sum(row["matched"] for row in rows)),
    "total_target": int(sum(row["target"] for row in rows)),
    "total_played": int(sum(row["played"] for row in rows)),
    "total_mispresses": int(sum(row["mispresses"] for row in rows)),
    "mean_event_f1": float(sum(row["event_f1"] for row in rows) / len(rows)),
    "mean_frame_f1": float(sum(row["frame_f1"] for row in rows) / len(rows)),
    "source_mean_event_f1": float(base["mean_event_f1"]),
    "source_mean_frame_f1": float(base["mean_frame_f1"]),
}
out["delta_mean_event_f1"] = float(out["mean_event_f1"] - out["source_mean_event_f1"])
(root / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
