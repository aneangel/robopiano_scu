#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J etu_m07s
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/etu_m07s_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/etu_m07s_%j.err
#SBATCH -t 05:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_07/trajectory.npz
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro07_etude_stable_20260523
mkdir -p "$ROOT"

"$PY" etude_contact_cem_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/settle5_open" \
  --samples-per-mask 100 \
  --elite-count 6 \
  --iterations 2 \
  --top-seeds 5 \
  --finger-sigma 0.050 \
  --forearm-sigma 0.035 \
  --min-mean-static-improvement 0.02 \
  --max-fp-increase 1.00 \
  --fp-rank-weight 0.05 \
  --settle-steps 5 \
  --threshold 0.5 \
  --seed 901

"$PY" etude_contact_cem_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/settle3_open" \
  --samples-per-mask 100 \
  --elite-count 6 \
  --iterations 2 \
  --top-seeds 5 \
  --finger-sigma 0.050 \
  --forearm-sigma 0.035 \
  --min-mean-static-improvement 0.02 \
  --max-fp-increase 1.00 \
  --fp-rank-weight 0.05 \
  --settle-steps 3 \
  --threshold 0.5 \
  --seed 902

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro07_etude_stable_20260523')
rows = []
for path in sorted(root.glob('*/summary.json')):
    row = json.loads(path.read_text())
    row['variant'] = path.parent.name
    row.pop('mask_rows', None)
    rows.append(row)
summary = {
    'root': str(root),
    'etude_open': {
        'event_f1': 0.182926829268293,
        'frame_f1': 0.412531214528944,
        'matched': 15,
        'target': 34,
        'mispresses': 115,
        'played': 130,
    },
    'rows': rows,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
