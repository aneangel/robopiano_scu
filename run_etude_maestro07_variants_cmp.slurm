#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J etu_m07
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/etu_m07_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/etu_m07_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_07/trajectory.npz
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro07_etude_variants_20260523
mkdir -p "$ROOT"

"$PY" etude_contact_cem_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/recall" \
  --samples-per-mask 120 \
  --elite-count 6 \
  --iterations 2 \
  --top-seeds 5 \
  --finger-sigma 0.040 \
  --forearm-sigma 0.026 \
  --min-mean-static-improvement 0.08 \
  --max-fp-increase 0.00 \
  --fp-rank-weight 0.15 \
  --threshold 0.5 \
  --seed 612

"$PY" etude_contact_cem_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/strict" \
  --samples-per-mask 120 \
  --elite-count 6 \
  --iterations 2 \
  --top-seeds 5 \
  --finger-sigma 0.030 \
  --forearm-sigma 0.018 \
  --min-mean-static-improvement 0.05 \
  --max-fp-increase -0.25 \
  --fp-rank-weight 0.25 \
  --threshold 0.5 \
  --seed 611

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro07_etude_variants_20260523')
rows = []
for path in sorted(root.glob('*/summary.json')):
    row = json.loads(path.read_text())
    row['variant'] = path.parent.name
    row.pop('mask_rows', None)
    rows.append(row)
summary = {
    'root': str(root),
    'capriccio_strict_reference': {
        'event_f1': 0.05303030303030303,
        'frame_f1': 0.238186056,
        'matched': 7,
        'target': 34,
        'mispresses': 223,
    },
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
