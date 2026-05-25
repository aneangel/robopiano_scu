#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J cap10agg
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cap10agg_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/cap10agg_%j.err
#SBATCH -t 00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_caprice_contact_20260524")
adaptive = json.loads(Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_tremolo_adaptive_20260524/summary.json").read_text())
by_run = {row["run"]: row for row in adaptive["runs"]}
rows = []
for idx in range(1, 11):
    run = f"maestro_{idx:02d}"
    path = root / run / "summary.json"
    if not path.exists():
        rows.append({"run": run, "missing": True})
        continue
    row = json.loads(path.read_text())
    source = by_run[run]
    final = row["final_score"]
    out = {
        "run": run,
        "event_f1": float(final["event_f1"]),
        "frame_f1": float(final["frame_f1"]),
        "matched": int(final["matched"]),
        "target": int(final["target"]),
        "played": int(final["played"]),
        "mispresses": int(final["mispresses"]),
        "accepted_count": int(row.get("accepted_count", 0)),
        "rejected_count": int(row.get("rejected_count", 0)),
        "eval_count": int(row.get("eval_count", 0)),
        "source_event_f1": float(source["event_f1"]),
        "source_frame_f1": float(source["frame_f1"]),
        "source_matched": int(source["matched"]),
        "source_mispresses": int(source["mispresses"]),
        "source_choice": source.get("source_choice"),
    }
    out["delta_event_f1"] = float(out["event_f1"] - out["source_event_f1"])
    out["delta_mispresses"] = int(out["mispresses"] - out["source_mispresses"])
    rows.append(out)

complete = [row for row in rows if not row.get("missing")]
summary = {
    "root": str(root),
    "source_policy": adaptive["policy"],
    "runs_completed": int(len(complete)),
    "runs": rows,
}
if complete:
    summary.update(
        {
            "mean_event_f1": float(sum(row["event_f1"] for row in complete) / len(complete)),
            "mean_frame_f1": float(sum(row["frame_f1"] for row in complete) / len(complete)),
            "total_matched": int(sum(row["matched"] for row in complete)),
            "total_target": int(sum(row["target"] for row in complete)),
            "total_played": int(sum(row["played"] for row in complete)),
            "total_mispresses": int(sum(row["mispresses"] for row in complete)),
        }
    )
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
