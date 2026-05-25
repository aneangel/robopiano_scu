#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J trem2rand
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/tremolo_second_random_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/tremolo_second_random_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_tremolo_finger_20260524
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/tremolo_second_random_20260524
mkdir -p "$ROOT"

idx=$(shuf -i 1-10 -n 1)
run=$(printf "maestro_%02d" "$idx")
out="$ROOT/$run"
echo "=== Tremolo second-pass random gate $run ==="

"$PY" tremolo_nearest_finger_scrub_gate.py \
  --source-npz "$SRC/$run/trajectory.npz" \
  --output-dir "$out" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --pre-dense-frames 1 \
  --post-dense-frames 8 \
  --top-fingers 2 \
  --modes neutral,hold_before,zero \
  --max-events 60 \
  --max-total-evals 180 \
  --allow-equal-f1-mispress-drop

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/tremolo_second_random_20260524")
source = json.loads(Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_tremolo_finger_20260524/summary.json").read_text())
source_by_run = {row["run"]: row for row in source["runs"]}
rows = []
for path in sorted(root.glob("maestro_*/summary.json")):
    row = json.loads(path.read_text())
    run = path.parent.name
    source_row = source_by_run[run]
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
        "eval_count": int(row.get("eval_count", 0)),
        "source_event_f1": float(source_row["event_f1"]),
        "source_frame_f1": float(source_row["frame_f1"]),
        "source_matched": int(source_row["matched"]),
        "source_mispresses": int(source_row["mispresses"]),
    }
    out["delta_event_f1"] = float(out["event_f1"] - out["source_event_f1"])
    out["delta_mispresses"] = int(out["mispresses"] - out["source_mispresses"])
    rows.append(out)
summary = {"root": str(root), "source_root": str(source["root"]), "runs": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
