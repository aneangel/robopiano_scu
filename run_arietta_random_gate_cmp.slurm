#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J arietrnd
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/arietta_random_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/arietta_random_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/arietta_random_gate_20260524
mkdir -p "$ROOT"

RUN=$("$PY" - <<'PY'
import random
runs = [f"maestro_{i:02d}" for i in range(1, 11)]
print(random.choice(runs))
PY
)
echo "$RUN" > "$ROOT/selected_run.txt"
echo "Selected random run: $RUN"

SOURCE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_legato_caprice_20260524/$RUN/trajectory.npz
if [ "$RUN" = "maestro_01" ] && [ -f /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro01_20260524/$RUN/trajectory.npz ]; then
  SOURCE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro01_20260524/$RUN/trajectory.npz
fi
if [ "$RUN" = "maestro_07" ] && [ -f /WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro07_20260524/$RUN/trajectory.npz ]; then
  SOURCE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/rondo_missed_maestro07_20260524/$RUN/trajectory.npz
fi

"$PY" arietta_lateral_margin_gate.py \
  --source-npz "$SOURCE" \
  --output-dir "$ROOT/$RUN" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pre-dense-frames 2 \
  --post-dense-frames 10 \
  --ty-deltas=-0.030,-0.024,-0.016,-0.010,-0.006,0.006,0.010,0.016,0.024,0.030 \
  --tx-deltas=0.0 \
  --active-target-only \
  --max-events 80 \
  --max-total-evals 180 \
  --allow-equal-f1-mispress-drop

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/arietta_random_gate_20260524")
run = (root / "selected_run.txt").read_text().strip()
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
