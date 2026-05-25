#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J legcap
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/legato_on_caprice_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/legato_on_caprice_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/legato_on_caprice_random_20260524
RUN=maestro_09
mkdir -p "$ROOT"

"$PY" legato_duplicate_hold_gate.py \
  --source-npz /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/caprice_contact_random_20260524/$RUN/trajectory.npz \
  --output-dir "$ROOT/$RUN" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pre-dense-frames 10 \
  --post-dense-frames 8 \
  --top-fingers 2 \
  --modes onset,onset_plus,pre_50ms,pre_100ms,mid_interval \
  --max-events 80 \
  --max-total-evals 180 \
  --allow-equal-f1-mispress-drop

"$PY" - <<'PY'
import json
from pathlib import Path

run = "maestro_09"
root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/legato_on_caprice_random_20260524")
caprice = json.loads(Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/caprice_contact_random_20260524/summary.json").read_text())["runs"][0]
row = json.loads((root / run / "summary.json").read_text())
final = row["final_score"]
out = {
    "root": str(root),
    "runs": [
        {
            "run": run,
            "event_f1": float(final["event_f1"]),
            "frame_f1": float(final["frame_f1"]),
            "matched": int(final["matched"]),
            "target": int(final["target"]),
            "played": int(final["played"]),
            "mispresses": int(final["mispresses"]),
            "accepted_count": int(row.get("accepted_count", 0)),
            "duplicate_events": int(row.get("duplicate_events", 0)),
            "eval_count": int(row.get("eval_count", 0)),
            "source_event_f1": float(caprice["event_f1"]),
            "source_frame_f1": float(caprice["frame_f1"]),
            "source_matched": int(caprice["matched"]),
            "source_mispresses": int(caprice["mispresses"]),
        }
    ],
}
out["runs"][0]["delta_event_f1"] = float(out["runs"][0]["event_f1"] - out["runs"][0]["source_event_f1"])
out["runs"][0]["delta_mispresses"] = int(out["runs"][0]["mispresses"] - out["runs"][0]["source_mispresses"])
(root / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
