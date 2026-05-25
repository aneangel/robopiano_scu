#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J caprand
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/caprice_random_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/caprice_random_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
TREM=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_tremolo_finger_20260524
FANDANGO_NEUTRAL=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/fandango_neutral_gate_20260524
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/caprice_contact_random_20260524
mkdir -p "$ROOT"

idx=$(shuf -i 1-10 -n 1)
run=$(printf "maestro_%02d" "$idx")
if [[ "$run" == "maestro_07" ]]; then
  source_npz="$FANDANGO_NEUTRAL/$run/hybrid/trajectory.npz"
else
  source_npz="$TREM/$run/trajectory.npz"
fi

echo "=== Caprice contact-aware random gate $run ==="
echo "source_npz=$source_npz"

"$PY" caprice_contact_scrub_gate.py \
  --source-npz "$source_npz" \
  --output-dir "$ROOT/$run" \
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
  --max-events 45 \
  --max-total-evals 160 \
  --allow-equal-f1-mispress-drop \
  --seed 524

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/caprice_contact_random_20260524")
adaptive = json.loads(Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_tremolo_adaptive_20260524/summary.json").read_text())
by_run = {row["run"]: row for row in adaptive["runs"]}
rows = []
for path in sorted(root.glob("maestro_*/summary.json")):
    row = json.loads(path.read_text())
    run = path.parent.name
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
        "eval_count": int(row.get("eval_count", 0)),
        "source_event_f1": float(source["event_f1"]),
        "source_frame_f1": float(source["frame_f1"]),
        "source_matched": int(source["matched"]),
        "source_mispresses": int(source["mispresses"]),
    }
    out["delta_event_f1"] = float(out["event_f1"] - out["source_event_f1"])
    out["delta_mispresses"] = int(out["mispresses"] - out["source_mispresses"])
    rows.append(out)
summary = {"root": str(root), "source_policy": adaptive["policy"], "runs": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
