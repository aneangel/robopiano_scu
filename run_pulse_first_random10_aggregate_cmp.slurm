#!/usr/bin/env bash
#SBATCH -p cmp
#SBATCH -J pulseag
#SBATCH -o /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/pulseag_%j.out
#SBATCH -e /WAVE/projects/ECEN-524-Wi26/robopiano/Impromptu/slurm_logs/pulseag_%j.err
#SBATCH -t 00:15:00
#SBATCH --cpus-per-task=1

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano
PY=/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python
"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np
root = Path('/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_pulse_first_20260523')
rows = []
missing = []
for idx in range(1, 11):
    run = f'maestro_{idx:02d}'
    path = root / run / 'summary.json'
    if not path.exists():
        missing.append(run)
        continue
    row = json.loads(path.read_text())
    row['run'] = run
    rows.append(row)
frame = np.asarray([float(r.get('frame_f1', 0.0)) for r in rows], dtype=np.float64)
event = np.asarray([float(r.get('event_f1', 0.0)) for r in rows], dtype=np.float64)
summary = {
    'root': str(root),
    'source_root': '/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/maestro_random10_etude_open_20260523',
    'runs_completed': int(len(rows)),
    'missing_runs': missing,
    'runs': rows,
    'mean_frame_f1': float(frame.mean()) if frame.size else 0.0,
    'median_frame_f1': float(np.median(frame)) if frame.size else 0.0,
    'mean_event_f1': float(event.mean()) if event.size else 0.0,
    'median_event_f1': float(np.median(event)) if event.size else 0.0,
    'total_matched': int(sum(int(r.get('matched', 0)) for r in rows)),
    'total_target': int(sum(int(r.get('target', 0)) for r in rows)),
    'total_mispresses': int(sum(int(r.get('mispresses', 0)) for r in rows)),
    'total_played': int(sum(int(r.get('played', 0)) for r in rows)),
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
if missing:
    raise SystemExit(f'Missing pulse summaries: {missing}')
PY
