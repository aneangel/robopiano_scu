#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J sch_m10
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sch_m10_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/sch_m10_%j.err
#SBATCH -t 03:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_10/trajectory.npz
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_scherzo_split_retrieval_20260523
rm -rf "$ROOT"
mkdir -p "$ROOT"

$PY scherzo_hand_split_retrieval_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/split44_empty_zoh" \
  --max-songs 160 \
  --demos-per-song 2 \
  --stride 2 \
  --split-key 44 \
  --min-side-frame-f1 0.95 \
  --dense-mode zoh \
  --replace-empty-side \
  --threshold 0.5

$PY scherzo_hand_split_retrieval_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/split48_empty_zoh" \
  --max-songs 160 \
  --demos-per-song 2 \
  --stride 2 \
  --split-key 48 \
  --min-side-frame-f1 0.95 \
  --dense-mode zoh \
  --replace-empty-side \
  --threshold 0.5

$PY scherzo_hand_split_retrieval_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/split44_active_zoh" \
  --max-songs 160 \
  --demos-per-song 2 \
  --stride 2 \
  --split-key 44 \
  --min-side-frame-f1 0.95 \
  --dense-mode zoh \
  --threshold 0.5

$PY scherzo_hand_split_retrieval_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/split44_empty_linear" \
  --max-songs 160 \
  --demos-per-song 2 \
  --stride 2 \
  --split-key 44 \
  --min-side-frame-f1 0.95 \
  --dense-mode linear \
  --replace-empty-side \
  --threshold 0.5

$PY - <<'PY'
import json
from pathlib import Path
root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro10_scherzo_split_retrieval_20260523')
rows = []
for path in sorted(root.glob('*/summary.json')):
    row = json.loads(path.read_text())
    row['variant'] = path.parent.name
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
    'capriccio_tight_best': {
        'event_f1': 0.2616822429906542,
        'frame_f1': 0.09546643417611159,
        'matched': 14,
        'target': 64,
        'mispresses': 29,
    },
    'rows': rows,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
