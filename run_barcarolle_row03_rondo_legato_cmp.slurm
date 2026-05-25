#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J barc03
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/barc03_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/barc03_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/barcarolle_row03_rondo_legato_20260524
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random30_best_adaptive_20260524/maestro30_03/rondo/trajectory.npz
LEGATO=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random30_best_adaptive_20260524/maestro30_03/legato_caprice/trajectory.npz
mkdir -p "$ROOT"

"$PY" barcarolle_event_window_ensemble_gate.py \
  --source-npz "$SRC" \
  --candidate-npz legato="$LEGATO" \
  --output-dir "$ROOT/maestro30_03" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --event-types missed \
  --scopes target_hand,full \
  --pre-dense-frames 2 \
  --post-dense-frames 10 \
  --max-events 120 \
  --max-total-evals 240 \
  --keep-if-matched-increases \
  --matched-gain-max-f1-drop 0.002 \
  --allow-equal-f1-mispress-drop

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/barcarolle_row03_rondo_legato_20260524')
base = json.loads(Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random30_best_adaptive_20260524/maestro30_03/best_summary.json').read_text())
row = json.loads((root / 'maestro30_03' / 'summary.json').read_text())
final = row['final_score']
out = {
    'root': str(root),
    'run': 'maestro30_03',
    'base_event_f1': float(base['event_f1']),
    'base_frame_f1': float(base['frame_f1']),
    'base_matched': int(base['matched']),
    'base_target': int(base['target']),
    'base_played': int(base['played']),
    'base_mispresses': int(base['mispresses']),
    'event_f1': float(final['event_f1']),
    'frame_f1': float(final['frame_f1']),
    'matched': int(final['matched']),
    'target': int(final['target']),
    'played': int(final['played']),
    'mispresses': int(final['mispresses']),
    'accepted_count': int(row.get('accepted_count', 0)),
    'eval_count': int(row.get('eval_count', 0)),
}
out['delta_event_f1'] = out['event_f1'] - out['base_event_f1']
out['delta_mispresses'] = out['mispresses'] - out['base_mispresses']
(root / 'summary.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
print(json.dumps(out, indent=2, sort_keys=True))
PY
