#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J sarabande_gate
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sarabande_gate_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sarabande_gate_%j.err
#SBATCH -t 01:30:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/sarabande_random_gate_20260524
mkdir -p "$ROOT"

idx=$(shuf -i 1-10 -n 1)
run=$(printf "maestro_%02d" "$idx")
echo "=== Sarabande random gate $run ==="
for variant in default lift all_fingers; do
  out="$ROOT/$run/$variant"
  extra=()
  if [[ "$variant" == "lift" ]]; then
    extra=(--lift-straight-anchors --straight-lift-height 0.020)
  elif [[ "$variant" == "all_fingers" ]]; then
    extra=(--straighten-all-fingers --lift-straight-anchors --straight-lift-height 0.020)
  fi
  "$PY" sarabande_straight_anchor_gate.py \
    --source-npz "$SRC/$run/trajectory.npz" \
    --output-dir "$out" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --release-fraction 0.20 \
    --approach-fraction 0.40 \
    "${extra[@]}"
done

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/sarabande_random_gate_20260524')
src = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523')
rows = []
for path in sorted(root.glob('maestro_*/*/summary.json')):
    data = json.loads(path.read_text())
    data['run'] = path.parts[-3]
    data['variant'] = path.parent.name
    old = json.loads((src / data['run'] / 'summary.json').read_text())
    data['old_event_f1'] = float(old.get('event_f1', 0.0))
    data['old_frame_f1'] = float(old.get('frame_f1', 0.0))
    data['old_matched'] = int(old.get('matched', 0))
    data['old_mispresses'] = int(old.get('mispresses', 0))
    data['delta_event_f1'] = float(data.get('event_f1', 0.0)) - data['old_event_f1']
    data['delta_frame_f1'] = float(data.get('frame_f1', 0.0)) - data['old_frame_f1']
    rows.append(data)
summary = {'root': str(root), 'source_root': str(src), 'runs': rows}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
