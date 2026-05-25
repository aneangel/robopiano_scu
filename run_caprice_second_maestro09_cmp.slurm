#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J cap2m09
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/caprice_second_m09_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/caprice_second_m09_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/caprice_second_maestro09_20260524
RUN=maestro_09
mkdir -p "$ROOT"

"$PY" caprice_contact_scrub_gate.py \
  --source-npz /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_caprice_contact_20260524/$RUN/trajectory.npz \
  --output-dir "$ROOT/$RUN" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pre-dense-frames 1 \
  --post-dense-frames 8 \
  --top-fingers 2 \
  --modes neutral,hold_before,zero \
  --samples-per-finger 18 \
  --top-candidates 4 \
  --finger-sigma 0.035 \
  --forearm-sigma 0.016 \
  --static-fp-weight 0.30 \
  --static-fn-weight 0.18 \
  --max-events 60 \
  --max-total-evals 200 \
  --allow-equal-f1-mispress-drop \
  --seed 525

"$PY" - <<'PY'
import json
from pathlib import Path

run = "maestro_09"
root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/caprice_second_maestro09_20260524")
source = json.loads(Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_caprice_contact_20260524/summary.json").read_text())
source_row = {row["run"]: row for row in source["runs"]}[run]
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
            "eval_count": int(row.get("eval_count", 0)),
            "source_event_f1": float(source_row["event_f1"]),
            "source_frame_f1": float(source_row["frame_f1"]),
            "source_matched": int(source_row["matched"]),
            "source_mispresses": int(source_row["mispresses"]),
        }
    ],
}
out["runs"][0]["delta_event_f1"] = float(out["runs"][0]["event_f1"] - out["runs"][0]["source_event_f1"])
out["runs"][0]["delta_mispresses"] = int(out["runs"][0]["mispresses"] - out["runs"][0]["source_mispresses"])
(root / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
