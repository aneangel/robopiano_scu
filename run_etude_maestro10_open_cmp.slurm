#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J etu_m10_o
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/etu_m10_o_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/etu_m10_o_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_10/trajectory.npz
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_etude_open_20260523
rm -rf "$ROOT"
mkdir -p "$ROOT"

"$PY" etude_contact_cem_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/open_recall" \
  --samples-per-mask 180 \
  --elite-count 8 \
  --iterations 3 \
  --top-seeds 6 \
  --finger-sigma 0.050 \
  --forearm-sigma 0.035 \
  --min-mean-static-improvement 0.02 \
  --max-fp-increase 1.00 \
  --fp-rank-weight 0.05 \
  --threshold 0.5 \
  --seed 713

"$PY" etude_contact_cem_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/mild_precision" \
  --samples-per-mask 180 \
  --elite-count 8 \
  --iterations 3 \
  --top-seeds 6 \
  --finger-sigma 0.045 \
  --forearm-sigma 0.030 \
  --min-mean-static-improvement 0.04 \
  --max-fp-increase 0.50 \
  --fp-rank-weight 0.10 \
  --threshold 0.5 \
  --seed 714

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_etude_open_20260523')
rows = []
for path in sorted(root.glob('*/summary.json')):
    row = json.loads(path.read_text())
    row['variant'] = path.parent.name
    row.pop('mask_rows', None)
    rows.append(row)
summary = {
    'root': str(root),
    'baseline_weighted': {
        'event_f1': 0.2476190476190476,
        'frame_f1': 0.09269785745518146,
        'matched': 13,
        'target': 64,
        'mispresses': 28,
        'played': 41,
    },
    'etude_recall_gate': {
        'event_f1': 0.30000000000000004,
        'frame_f1': 0.18637399047421827,
        'matched': 18,
        'target': 64,
        'mispresses': 38,
        'played': 56,
    },
    'rows': rows,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
