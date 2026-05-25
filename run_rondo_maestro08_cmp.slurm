#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J rondom08
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/rondo_m08_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/rondo_m08_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro08_20260524
RUN=maestro_08
mkdir -p "$ROOT"

"$PY" rondo_missed_event_inject_gate.py \
  --source-npz /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/fandango_neutral_gate_20260524/$RUN/hybrid/trajectory.npz \
  --candidate-npz /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_tremolo_finger_20260524/$RUN/trajectory.npz \
  --output-dir "$ROOT/$RUN" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pre-dense-frames 1 \
  --post-dense-frames 10 \
  --top-fingers 2 \
  --scopes finger,hand,full \
  --max-events 80 \
  --max-total-evals 180 \
  --keep-if-matched-increases \
  --matched-gain-max-f1-drop 0.002

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro08_20260524")
run = "maestro_08"
base_path = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_adaptive_legato_caprice_rondo_20260524/summary.json")
base = json.loads(base_path.read_text())
base_row = {row["run"]: row for row in base["runs"]}[run]
row = json.loads((root / run / "summary.json").read_text())
final = row["final_score"]
out = {
    "root": str(root),
    "run": run,
    "base_summary": str(base_path),
    "base_event_f1": float(base_row["event_f1"]),
    "base_frame_f1": float(base_row["frame_f1"]),
    "base_matched": int(base_row["matched"]),
    "base_target": int(base_row["target"]),
    "base_played": int(base_row["played"]),
    "base_mispresses": int(base_row["mispresses"]),
    "event_f1": float(final["event_f1"]),
    "frame_f1": float(final["frame_f1"]),
    "matched": int(final["matched"]),
    "target": int(final["target"]),
    "played": int(final["played"]),
    "mispresses": int(final["mispresses"]),
    "accepted_count": int(row.get("accepted_count", 0)),
    "eval_count": int(row.get("eval_count", 0)),
}
out["delta_event_f1"] = float(out["event_f1"] - out["base_event_f1"])
out["delta_mispresses"] = int(out["mispresses"] - out["base_mispresses"])
(root / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
