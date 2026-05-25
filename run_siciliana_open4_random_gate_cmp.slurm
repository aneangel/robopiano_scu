#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J sic_o4
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sic_o4_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sic_o4_%j.err
#SBATCH -t 01:20:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/siciliana_open4_random_gate_20260524
mkdir -p "$ROOT"

idx=$(shuf -i 1-10 -n 1)
run=$(printf "maestro_%02d" "$idx")
echo "=== Siciliana open4 random gate $run ==="
for variant in fp0 fp1; do
  out="$ROOT/$run/$variant"
  extra=()
  if [[ "$variant" == "fp0" ]]; then
    extra=(--max-static-fp 0 --max-pulses-per-onset 4 --played-weight 0.04)
  else
    extra=(--max-static-fp 1 --max-pulses-per-onset 4 --played-weight 0.035)
  fi
  "$PY" siciliana_grouped_injection_gate.py \
    --source-npz "$SRC/$run/trajectory.npz" \
    --output-dir "$out" \
    --environment-name RoboPianist-debug-NocturneRousseau-v0 \
    --threshold 0.5 \
    --pulse-width 1 \
    --pulse-gap 0 \
    --max-candidate-played-count 0 \
    "${extra[@]}"
done

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/siciliana_open4_random_gate_20260524')
source = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523')
rows = []
for path in sorted(root.glob('maestro_*/*/summary.json')):
    row = json.loads(path.read_text())
    row['run'] = path.parts[-3]
    row['variant'] = path.parts[-2]
    old = json.loads((source / row['run'] / 'summary.json').read_text())
    row['old_event_f1'] = float(old.get('event_f1', 0.0))
    row['old_frame_f1'] = float(old.get('frame_f1', 0.0))
    row['old_matched'] = int(old.get('matched', 0))
    row['old_mispresses'] = int(old.get('mispresses', 0))
    row['old_played'] = int(old.get('played', 0))
    row['delta_event_f1'] = float(row.get('event_f1', 0.0)) - row['old_event_f1']
    row['delta_frame_f1'] = float(row.get('frame_f1', 0.0)) - row['old_frame_f1']
    rows.append(row)
summary = {'root': str(root), 'source_root': str(source), 'runs': rows}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
