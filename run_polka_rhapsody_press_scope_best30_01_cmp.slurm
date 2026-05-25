#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J polkar01
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/polkar01_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/polkar01_%j.err
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano

PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/polka_rhapsody_press_scope_best30_01_20260524
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/waltz_contact_mpc_second_best30_01_20260524/maestro30_01/trajectory.npz
RHAPSODY=/WAVE/datasets/ccoelho_lab-jlanders/Rhapsody/random10_heldout2_prev_random_scale6_refine16_20260522/rhapsody_rpik.pt
mkdir -p "$ROOT"

"$PY" polka_chord_mpc_gate.py \
  --source-npz "$SRC" \
  --output-dir "$ROOT/maestro30_01" \
  --environment-name RoboPianist-debug-NocturneRousseau-v0 \
  --threshold 0.5 \
  --key-source press \
  --scopes finger,hand,full \
  --rhapsody-checkpoint "$RHAPSODY" \
  --rhapsody-refinement-steps 16 \
  --pre-dense-frames 1 \
  --post-dense-frames 8 \
  --max-events 20 \
  --max-patterns 48 \
  --max-total-evals 180 \
  --keep-if-matched-increases \
  --matched-gain-max-f1-drop 0.003 \
  --allow-equal-f1-mispress-drop

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/polka_rhapsody_press_scope_best30_01_20260524')
base = json.loads(Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/waltz_contact_mpc_second_best30_01_20260524/summary.json').read_text())
row = json.loads((root / 'maestro30_01' / 'summary.json').read_text())
final = row['final_score']
out = {
    'root': str(root),
    'run': 'maestro30_01',
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
