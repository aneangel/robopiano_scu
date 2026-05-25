#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J m07_puls
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/m07_puls_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/m07_puls_%j.err
#SBATCH -t 01:30:00
#SBATCH --cpus-per-task=2

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
export MUJOCO_GL=egl
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
SRC=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523/maestro_07/trajectory.npz
BASE=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_ikaware_weighted_20260522/maestro_07/trajectory.npz
ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro07_pulse_gate_20260523
mkdir -p "$ROOT"

"$PY" pulse_event_postprocess_gate.py --source-npz "$SRC" --baseline-npz "$BASE" --output-dir "$ROOT/post1_baseline" --post-frames 1 --rest-source baseline --threshold 0.5
"$PY" pulse_event_postprocess_gate.py --source-npz "$SRC" --baseline-npz "$BASE" --output-dir "$ROOT/post2_baseline" --post-frames 2 --rest-source baseline --threshold 0.5
"$PY" pulse_event_postprocess_gate.py --source-npz "$SRC" --baseline-npz "$BASE" --output-dir "$ROOT/post1_first" --post-frames 1 --rest-source first --threshold 0.5

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro07_pulse_gate_20260523')
rows = []
for path in sorted(root.glob('*/summary.json')):
    row = json.loads(path.read_text())
    row['variant'] = path.parent.name
    rows.append(row)
summary = {'root': str(root), 'etude_open': {'event_f1': 0.182926829268293, 'matched': 15, 'target': 34, 'mispresses': 115, 'played': 130}, 'rows': rows}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
